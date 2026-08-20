"""Train the geometry-guided V-JEPA 2.1 ProjectedPredictor.

The model starts from the deterministic latent proposal produced by geometric
RGB warping plus copy-forward infill, then predicts a learned correction in the
frozen V-JEPA 2.1 ViT-B image-latent space::

    Z_hat = Z0 + DeltaZ

Training uses the four future horizons produced by ``ProjectedPredictor``.  The
loss deliberately mirrors the simple dense V-JEPA 2.1 decomposition without
its distance weighting:

    L = L_context + L_masked

where both terms are independently normalised L1 losses in the frozen 768-D
ViT-B image-latent space:

    L_context = mean_{geometrically valid patches} |Z_hat - Z*|
    L_masked  = mean_{geometrically missing patches} |Z_hat - Z*|

There is no latent normalisation, no context-distance weighting and no separate
teacher-forcing objective.  The four-step latent rollout is already performed
inside ``ProjectedPredictor.forward``.

KITTI is 10 Hz.  Use ``--dt 0.5`` (frame stride 5) or ``--dt 0.2`` (frame
stride 2).  The default iteration budget and validation cadence match
``train_ac_predictor.py``: 80k micro-iterations, gradient accumulation of 4,
and validation every 5k iterations.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from torch import Tensor

from jepa_drive_wm.data.data_interface_projected_predictor import (
    KITTIProjectedPredictorLoaders,
    projected_predictor_inputs,
)
from jepa_drive_wm.data.splits import SPLIT_V1
from jepa_drive_wm.models.predictors.projected_predictor.projected_predictor import (
    ProjectedPredictor,
    ProjectedPredictorOutput,
)
from jepa_drive_wm.models.predictors.projected_predictor.warp_rgb import (
    DEFAULT_PATCH_COVERAGE_THRESHOLD,
    DEFAULT_RADIUS_PX,
    WarpModule,
)
from jepa_drive_wm.models.predictors.projected_predictor.released_vitb import (
    build_released_vitb_384,
)
from jepa_drive_wm.paths import OUTPUTS_DIR, VJEPA_CKPT_DIR
from jepa_drive_wm.training_utils import (
    autocast,
    build_warmup_cosine,
    infinite,
    init_run,
)


IMAGE_SIZE = (384, 1248)
PATCH_SIZE = 16
DEFAULT_VJEPA_CHECKPOINT = VJEPA_CKPT_DIR / "vjepa2_1_vitb_dist_vitG_384.pt"


@dataclass
class Losses:
    """Simple projected-predictor loss decomposition."""

    total: Tensor
    context: Tensor
    masked: Tensor
    per_horizon: Tensor  # [4], unpartitioned L1 used only for logging


def dt_to_frame_stride(dt: float) -> int:
    """Map the supported physical prediction steps to raw KITTI frame stride."""
    mapping = {0.2: 2, 0.5: 5}
    for supported_dt, stride in mapping.items():
        if abs(float(dt) - supported_dt) < 1e-8:
            return stride
    raise ValueError("ProjectedPredictor currently supports only dt=0.2 or dt=0.5 s.")


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean selected values, returning a differentiable zero if region is empty."""
    mask = mask.to(device=values.device, dtype=torch.bool)
    if mask.any():
        return values[mask].mean()
    return values.sum() * 0.0


def projected_predictor_loss(output: ProjectedPredictorOutput) -> Losses:
    """L1 in frozen ViT-B image-latent space, split by geometric validity.

    The two regions are independently normalised.  This prevents the usually
    larger geometrically valid/context region from numerically overwhelming the
    masked-patch completion objective.

    No distance weighting is used on context patches: every valid patch counts
    equally.  Loss arithmetic is explicitly fp32 even when the forward pass is
    under bf16 autocast.
    """
    if output.target_latents is None:
        raise ValueError(
            "ProjectedPredictorOutput has no target_latents. "
            "Training forward passes must supply future_rgb."
        )

    pred = output.predictions.float()          # [1,4,N,768]
    target = output.target_latents.float()     # [1,4,N,768]
    valid = output.patch_valid.to(dtype=torch.bool)  # [1,4,N]

    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {tuple(pred.shape)} vs "
            f"{tuple(target.shape)}."
        )
    if valid.shape != pred.shape[:-1]:
        raise ValueError(
            f"patch_valid shape {tuple(valid.shape)} does not match predictions "
            f"{tuple(pred.shape[:-1])}."
        )

    # Mean over the 768 latent channels first; then independently average each
    # spatial region across all four future horizons.
    patch_l1 = (pred - target).abs().mean(dim=-1)  # [1,4,N]
    context = _masked_mean(patch_l1, valid)
    masked = _masked_mean(patch_l1, ~valid)
    total = context + masked

    # Simple horizon health metric, not an extra optimisation term.
    per_horizon = patch_l1.mean(dim=(0, 2))  # [4]
    return Losses(total=total, context=context, masked=masked, per_horizon=per_horizon)


def _set_batch_intrinsics(
    model: ProjectedPredictor,
    batch: dict[str, Tensor],
) -> None:
    """Update the warper's registered K from this sample's resized calibration.

    The current ProjectedPredictor owns one WarpModule instance, while KITTI
    calibration can differ by sequence.  The data interface exposes the scaled
    per-sample intrinsics, so copy them into the warper before every forward.
    """
    intrinsics = batch["intrinsics"]
    if intrinsics.shape != (1, 3, 3):
        raise ValueError(
            f"Expected batch-size-one intrinsics [1,3,3], got {tuple(intrinsics.shape)}."
        )

    K = intrinsics[0].to(
        device=model.warp_module.K.device,
        dtype=model.warp_module.K.dtype,
    )
    with torch.no_grad():
        model.warp_module.K.copy_(K)


def batch_losses(
    model: ProjectedPredictor,
    batch: dict[str, Tensor],
    device: torch.device,
) -> Losses:
    """Run one projected-predictor sample and return the simple region losses."""
    _set_batch_intrinsics(model, batch)

    # The current model is deliberately per-sample because geometry gives
    # variable-length V-JEPA masks.  The helper removes the DataLoader's B=1
    # axis.  Individual model components move tensors to their required device.
    model_inputs = projected_predictor_inputs(batch)

    with autocast(device):
        output = model(**model_inputs)

    # Keep the loss definition outside autocast and explicitly in fp32.
    return projected_predictor_loss(output)


@torch.no_grad()
def evaluate(
    model: ProjectedPredictor,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Full validation pass; mean total/context/masked and per-horizon L1."""
    model.eval()

    total_sum = 0.0
    context_sum = 0.0
    masked_sum = 0.0
    horizon_sum = torch.zeros(4, dtype=torch.float64)
    n = 0

    for batch in loader:
        losses = batch_losses(model, batch, device)
        total_sum += losses.total.item()
        context_sum += losses.context.item()
        masked_sum += losses.masked.item()
        horizon_sum += losses.per_horizon.detach().cpu().double()
        n += 1

    n = max(n, 1)
    metrics = {
        "total": total_sum / n,
        "context": context_sum / n,
        "masked": masked_sum / n,
    }
    for k in range(4):
        metrics[f"h{k + 1}"] = float(horizon_sum[k] / n)
    return metrics


def build_model(
    *,
    device: torch.device,
    vjepa_checkpoint: Path,
    radius_px: float = DEFAULT_RADIUS_PX,
    patch_coverage_threshold: float = DEFAULT_PATCH_COVERAGE_THRESHOLD,
    finetune_predictor: bool = True,
) -> ProjectedPredictor:
    """Load pretrained ViT-B encoder/predictor and attach geometry + new head."""
    encoder, predictor = build_released_vitb_384(vjepa_checkpoint, device=device)

    # K is overwritten from every dataset sample before forward().  Identity is
    # merely a valid placeholder required to construct the WarpModule.
    warper = WarpModule(
        intrinsics=torch.eye(3, dtype=torch.float32),
        image_size=IMAGE_SIZE,
        radius_px=radius_px,
        patch_size=PATCH_SIZE,
        patch_coverage_threshold=patch_coverage_threshold,
    ).to(device)

    model = ProjectedPredictor(
        encoder=encoder,
        predictor=predictor,
        warp_module=warper,
        patch_size=PATCH_SIZE,
        image_dim=768,
        predictor_dim=384,
        finetune_predictor=finetune_predictor,
        zero_init_correction=True,
    ).to(device)
    return model


def _optimizer(
    model: ProjectedPredictor,
    *,
    head_lr: float,
    predictor_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW with a smaller LR for the pretrained predictor body.

    The predictor's mask tokens join the head-LR group: the released
    checkpoint's eight mask tokens are degenerate/interchangeable, so the
    completion query is effectively a freshly initialised parameter rather
    than pretrained structure worth protecting.
    """
    mask_token_ids = {id(p) for p in model.predictor.mask_tokens.parameters()}

    head_params = [p for p in model.correction_head.parameters() if p.requires_grad]
    head_params += [
        p for p in model.predictor.mask_tokens.parameters() if p.requires_grad
    ]
    predictor_params = [
        p
        for p in model.predictor.parameters()
        if p.requires_grad and id(p) not in mask_token_ids
    ]

    groups = [{"params": head_params, "lr": head_lr, "name": "head_and_mask_tokens"}]
    if predictor_params:
        groups.append(
            {"params": predictor_params, "lr": predictor_lr, "name": "predictor"}
        )

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _save_checkpoint(
    path: Path,
    *,
    model: ProjectedPredictor,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    val_metrics: dict[str, float],
    step_seconds: float,
    frame_stride: int,
    vjepa_checkpoint: Path,
) -> None:
    """Save learned state without duplicating the frozen ViT-B encoder weights."""
    torch.save(
        {
            # Predictor includes the post-trained pretrained body.  The frozen
            # obsolete 1664-D heads are harmlessly included for a simple load.
            "predictor": model.predictor.state_dict(),
            "correction_head": model.correction_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "val": val_metrics,
            "step_seconds": step_seconds,
            "frame_stride": frame_stride,
            "base_vjepa_checkpoint": str(vjepa_checkpoint),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the geometry-guided V-JEPA2.1 ProjectedPredictor"
    )

    # Match train_ac_predictor.py's optimisation budget/cadence.
    parser.add_argument("--max-iters", type=int, default=80_000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=5_000)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--log-every", type=int, default=100)

    # Current ProjectedPredictor is intentionally per-sample.
    parser.add_argument("--batch-size", type=int, default=1, choices=[1])
    parser.add_argument("--num-workers", type=int, default=8)

    # Physical horizon: KITTI is 10 Hz -> dt 0.2 = stride 2, dt 0.5 = stride 5.
    parser.add_argument("--dt", type=float, default=0.5, choices=[0.2, 0.5])

    # The new head and the (degenerate) mask tokens learn at full rate; the
    # pretrained predictor body gets a 3x-lower default LR as a middle ground
    # between preserving its prior and letting it genuinely adapt (the
    # zero-init head already gates body gradients to zero at step 0).
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="384->768 head + mask-token LR"
    )
    parser.add_argument(
        "--predictor-lr",
        type=float,
        default=1e-4,
        help="pretrained predictor-body LR",
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--freeze-predictor",
        action="store_true",
        help="Train only the new 384->768 correction head.",
    )

    # Geometry defaults are defined once in warp_rgb.py; override per run only
    # for experiments.
    parser.add_argument("--radius-px", type=float, default=DEFAULT_RADIUS_PX)
    parser.add_argument(
        "--patch-coverage-threshold",
        type=float,
        default=DEFAULT_PATCH_COVERAGE_THRESHOLD,
    )

    parser.add_argument(
        "--vjepa-checkpoint",
        type=Path,
        default=DEFAULT_VJEPA_CHECKPOINT,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a few train batches and one validation batch, then exit.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_stride = dt_to_frame_stride(args.dt)

    loaders = KITTIProjectedPredictorLoaders(
        split=SPLIT_V1,
        frame_stride=frame_stride,
        image_size=IMAGE_SIZE,
        batch_size=1,
        num_workers=args.num_workers,
    )
    print(loaders)

    # Data interface is the source of truth for the realised physical step.
    if abs(loaders.step_seconds - args.dt) > 1e-8:
        raise RuntimeError(
            f"Requested dt={args.dt}, but dataset reports {loaders.step_seconds}s."
        )

    model = build_model(
        device=device,
        vjepa_checkpoint=args.vjepa_checkpoint,
        radius_px=args.radius_px,
        patch_coverage_threshold=args.patch_coverage_threshold,
        finetune_predictor=not args.freeze_predictor,
    )

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    predictor_trainable = sum(
        p.numel() for p in model.predictor.parameters() if p.requires_grad
    )
    head_trainable = sum(
        p.numel() for p in model.correction_head.parameters() if p.requires_grad
    )
    print(f"parameters: total={total_parameters / 1e6:.1f}M, "
          f"trainable={trainable_parameters / 1e6:.1f}M")
    print(f"  predictor trainable: {predictor_trainable / 1e6:.1f}M")
    print(f"  correction head:     {head_trainable / 1e6:.3f}M")

    step_tag = f"dt{loaders.step_seconds:g}s"
    if args.checkpoint is None:
        args.checkpoint = (
            OUTPUTS_DIR
            / "checkpoints_projected_predictor"
            / f"projected_predictor_{step_tag}.pt"
        )
    print(f"checkpoint: {args.checkpoint}")

    optimizer = _optimizer(
        model,
        head_lr=args.lr,
        predictor_lr=args.predictor_lr,
        weight_decay=args.weight_decay,
    )

    init_run(
        job_type="projected_predictor",
        name=(
            f"projected-{step_tag}-s{frame_stride}-"
            f"hlr{args.lr:g}-plr{args.predictor_lr:g}-"
            f"{args.max_iters // 1000}k"
        ),
        tags=["projected-predictor", "vjepa2.1", "geometry", "rollout", step_tag],
        config={
            **vars(args),
            "checkpoint": str(args.checkpoint),
            "vjepa_checkpoint": str(args.vjepa_checkpoint),
            "frame_stride": frame_stride,
            "step_seconds": loaders.step_seconds,
            "effective_batch": args.grad_accum,  # physical B=1
            "total_parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "loss": "L1_context + L1_masked",
        },
        split=SPLIT_V1,
        mode="disabled" if args.smoke_test else "online",
    )

    if args.smoke_test:
        from itertools import islice

        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch in islice(loaders.train, 2):
            losses = batch_losses(model, batch, device)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print(
                f"smoke train | total {losses.total.item():.4f} "
                f"context {losses.context.item():.4f} "
                f"masked {losses.masked.item():.4f}"
            )

        # One validation sample is enough for a smoke test; a complete V-JEPA
        # validation pass is intentionally avoided here.
        model.eval()
        val_batch = next(iter(loaders.validation))
        val_losses = batch_losses(model, val_batch, device)
        print(
            f"smoke val   | total {val_losses.total.item():.4f} "
            f"context {val_losses.context.item():.4f} "
            f"masked {val_losses.masked.item():.4f}"
        )
        print("smoke test passed: projected-predictor train/val forward+backward works.")
        wandb.finish()
        return

    total_opt_steps = args.max_iters // args.grad_accum
    scheduler = build_warmup_cosine(
        optimizer,
        total_opt_steps,
        int(args.warmup_frac * total_opt_steps),
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    # With the zero-initialised correction head, Z_hat == Z0 exactly, so this
    # pass records the deterministic warp+copy baseline (B1) under the
    # identical data/loss pipeline before any learning.
    baseline = evaluate(model, loaders.validation, device)
    model.train()
    print(
        f"  [val] iter 0 (B1 warp+copy baseline) | total {baseline['total']:.4f} "
        f"ctx {baseline['context']:.4f} mask {baseline['masked']:.4f} | "
        f"h1 {baseline['h1']:.4f} h2 {baseline['h2']:.4f} "
        f"h3 {baseline['h3']:.4f} h4 {baseline['h4']:.4f}",
        flush=True,
    )
    wandb.log({"iter": 0, **{f"val/{k}": v for k, v in baseline.items()}})
    wandb.summary["baseline_val_total"] = baseline["total"]

    train_batches = infinite(loaders.train)
    running_total = 0.0
    running_context = 0.0
    running_masked = 0.0
    running_horizon = torch.zeros(4, dtype=torch.float64)

    optimizer.zero_grad(set_to_none=True)
    model.train()
    start = time.time()

    for iteration in range(1, args.max_iters + 1):
        batch = next(train_batches)
        losses = batch_losses(model, batch, device)

        (losses.total / args.grad_accum).backward()

        grad_norm = None
        if iteration % args.grad_accum == 0:
            trainable = [p for p in model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_total += losses.total.item()
        running_context += losses.context.item()
        running_masked += losses.masked.item()
        running_horizon += losses.per_horizon.detach().cpu().double()

        if iteration % args.log_every == 0:
            rate = iteration / (time.time() - start)
            denom = float(args.log_every)
            log = {
                "iter": iteration,
                "train/total": running_total / denom,
                "train/context": running_context / denom,
                "train/masked": running_masked / denom,
                "train/lr_head": scheduler.get_last_lr()[0],
            }
            if len(scheduler.get_last_lr()) > 1:
                log["train/lr_predictor"] = scheduler.get_last_lr()[1]
            for k in range(4):
                log[f"train/h{k + 1}"] = float(running_horizon[k] / denom)
            if grad_norm is not None:
                log["train/grad_norm"] = grad_norm.item()

            print(
                f"iter {iteration:6d}/{args.max_iters} | "
                f"total {log['train/total']:.4f} "
                f"ctx {log['train/context']:.4f} "
                f"mask {log['train/masked']:.4f} | "
                f"lr {log['train/lr_head']:.2e} | {rate:.2f} it/s",
                flush=True,
            )
            wandb.log(log)

            running_total = 0.0
            running_context = 0.0
            running_masked = 0.0
            running_horizon.zero_()

        if iteration % args.val_every == 0 or iteration == args.max_iters:
            val = evaluate(model, loaders.validation, device)
            model.train()

            print(
                f"  [val] iter {iteration} | total {val['total']:.4f} "
                f"ctx {val['context']:.4f} mask {val['masked']:.4f} | "
                f"h1 {val['h1']:.4f} h2 {val['h2']:.4f} "
                f"h3 {val['h3']:.4f} h4 {val['h4']:.4f}",
                flush=True,
            )
            wandb.log(
                {
                    "iter": iteration,
                    "val/total": val["total"],
                    "val/context": val["context"],
                    "val/masked": val["masked"],
                    "val/h1": val["h1"],
                    "val/h2": val["h2"],
                    "val/h3": val["h3"],
                    "val/h4": val["h4"],
                }
            )

            # Total = equally weighted context + masked losses, so this is the
            # checkpoint criterion that exactly matches the optimisation target.
            if val["total"] < best_val:
                best_val = val["total"]
                _save_checkpoint(
                    args.checkpoint,
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    val_metrics=val,
                    step_seconds=loaders.step_seconds,
                    frame_stride=frame_stride,
                    vjepa_checkpoint=args.vjepa_checkpoint,
                )
                wandb.summary["best_val_total"] = best_val
                wandb.summary["best_iter"] = iteration
                print(f"          saved checkpoint (val total {best_val:.4f})")

    wandb.finish()


if __name__ == "__main__":
    main()
