"""
Train the dense depth decoder (DepthDecoder) on KITTI FoundationStereo pseudolabels.

Pipeline per sample (batch_size=1, frames indexed per sequence):

    V-JEPA grid [1, C, H, W]
        -> DepthDecoder (4x repeated final features -> DINOv3 DPT -> depth bins)
        -> depth [1, 1, h, w]
        -> bilinear upsample to the pseudolabel resolution
        -> loss against FoundationStereo depth [1, H_img, W_img]

Losses follow https://arxiv.org/pdf/2604.26454:

  1. Scale-invariant log loss (SiLog), lambda = 0.85:
         g_n   = log(pred_n) - log(gt_n)          over valid pixels
         L_slog = sqrt( mean(g^2) - lambda * mean(g)^2 )

  2. Hierarchical normalization loss (HN), grid scales M in {1, 4, 8}:
     the image is tiled into an M x M grid; within each tile, predicted and
     ground-truth depth are normalized (median / mean-absolute-deviation, in
     log space); the per-pixel loss is |N(gt) - N(pred)| averaged over every
     patch containing that pixel (tiles are disjoint, so: one patch per scale),
     then averaged over all valid pixels.

  3. Multi-scale gradient matching loss (MiDaS, arXiv 1907.01341 eq. 11),
     weight GRAD_WEIGHT: L1 on the gradients of the log-depth residual at
     dyadic strides {1, 2, 4, 8}. SiLog and HN are per-pixel, so a blurred
     boundary costs them little; this term charges it the full log-depth
     discontinuity.

The best model (lowest validation loss) is saved to CHECKPOINT_PATH.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

from jepa_drive_wm.data.data_interface_dense import KITTIDenseLoaders
from jepa_drive_wm.data.splits import SPLIT_V1
from jepa_drive_wm.models.dense_decoders.depth_decoder import DepthDecoder
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.training_utils import autocast, build_warmup_cosine, infinite

# ----------------------------------------------------------------------------- config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Depth validity range; must match FeaturesToDepth(min_depth=0.5, max_depth=80.0).
MIN_DEPTH = 0.5
MAX_DEPTH = 80.0

SILOG_LAMBDA = 0.85          # balancing parameter from the paper
HN_SCALES = (1, 4, 8)        # M: image tiled into an M x M grid per scale
HN_WEIGHT = 1.0              # total = silog + HN_WEIGHT * hn + GRAD_WEIGHT * grad
HN_MIN_VALID_PER_TILE = 32   # skip tiles with fewer valid pixels (stats too noisy)
GRAD_SCALES = 4              # dyadic strides 1..8; the stride-8 grid ~ matches the finest HN tile
GRAD_WEIGHT = 0.5            # MiDaS default alpha

# Iteration budget, not epochs: the proven DPT recipe trained a fixed number of
# forward/backward passes with a cosine schedule (see training_utils). MAX_ITERS
# counts micro-steps; GRAD_ACCUM of them make one effective-batch-8 optimizer step.
MAX_ITERS = 50_000
GRAD_ACCUM = 8               # effective batch size = GRAD_ACCUM * (loader batch 1)
VAL_EVERY = 5_000            # micro-steps between validation + checkpoint
LOG_EVERY = 100
NUM_WORKERS = 8              # dedicated training box; more workers hide shared-FS read latency
WARMUP_FRAC = 0.03           # fraction of optimizer steps spent warming the LR up
LEARNING_RATE = 3e-4         # peak LR (matches the old dpt_bins run's peak_lr)
WEIGHT_DECAY = 0.01

# All of the above are defaults; every one is overridable on the command line
# (see main's argparse) so budgets can be tuned per-machine without editing files.

# Sequence assignment lives in data/splits.py. These names exist so the wandb
# run config records the split, and evals/depth_evaluator.py imports
# TEST_SEQUENCES from here.
TRAIN_SEQUENCES = list(SPLIT_V1.train_sequences)
VALIDATION_SEQUENCES = list(SPLIT_V1.validation_sequences)
TEST_SEQUENCES = list(SPLIT_V1.test_sequences)

CHECKPOINT_PATH = OUTPUTS_DIR / "checkpoints_depth" / "depth_decoder_best.pt"


# ----------------------------------------------------------------------------- losses

def _valid_mask(target: torch.Tensor) -> torch.Tensor:
    """Pixels where the pseudolabel is trustworthy and inside the depth range."""
    return torch.isfinite(target) & (target > MIN_DEPTH) & (target < MAX_DEPTH)


def _loss_silog(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Scale-invariant log loss over valid pixels.

    pred, target, valid: [H, W] (single frame). pred must be > 0, which
    FeaturesToDepth guarantees (weighted sum of positive bins).
    """
    g = torch.log(pred[valid]) - torch.log(target[valid])
    # sqrt(mean(g^2) - lambda * mean(g)^2); small eps keeps sqrt differentiable at 0.
    return torch.sqrt((g**2).mean() - SILOG_LAMBDA * g.mean() ** 2 + 1e-8)


def _normalize_in_tile(x: torch.Tensor) -> torch.Tensor:
    """
    Median / mean-absolute-deviation normalization of a 1D tensor of log depths.
    This is the standard scale-and-shift-invariant normalization; applying it
    per tile is what makes the loss "hierarchical".
    """
    median = x.median()
    mad = (x - median).abs().mean().clamp_min(1e-6)
    return (x - median) / mad


def _loss_hn(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Hierarchical normalization loss over scales HN_SCALES.

    For each scale M the frame is tiled into an M x M grid. Within each tile,
    log(pred) and log(gt) are normalized independently and compared with L1.
    Every pixel accumulates one loss term per scale (tiles are disjoint), so we
    keep per-pixel sums and counts, average per pixel, then over valid pixels.
    """
    height, width = target.shape
    loss_sum = torch.zeros_like(target)
    loss_count = torch.zeros_like(target)

    log_pred = torch.log(pred)
    log_target = torch.where(valid, target, torch.ones_like(target)).log()

    for m in HN_SCALES:
        # Tile boundaries; the last tile absorbs the remainder pixels.
        row_edges = torch.linspace(0, height, m + 1).round().long()
        col_edges = torch.linspace(0, width, m + 1).round().long()

        for i in range(m):
            for j in range(m):
                rows = slice(row_edges[i], row_edges[i + 1])
                cols = slice(col_edges[j], col_edges[j + 1])

                tile_valid = valid[rows, cols]
                if tile_valid.sum() < HN_MIN_VALID_PER_TILE:
                    continue

                n_gt = _normalize_in_tile(log_target[rows, cols][tile_valid])
                n_pred = _normalize_in_tile(log_pred[rows, cols][tile_valid])

                tile_loss = torch.zeros_like(tile_valid, dtype=pred.dtype)
                tile_loss[tile_valid] = (n_gt - n_pred).abs()
                loss_sum[rows, cols] += tile_loss
                loss_count[rows, cols] += tile_valid.float()

    counted = valid & (loss_count > 0)
    if not counted.any():
        return pred.new_zeros(())
    return (loss_sum[counted] / loss_count[counted]).mean()


def _loss_grad(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Multi-scale gradient matching loss (MiDaS, arXiv 1907.01341 eq. 11) on log
    depth, where a global scale error is a constant offset and vanishes under
    differencing.

    The residual log(pred) - log(gt) is zeroed at invalid pixels; each scale is
    a strided subsample (pooling would fabricate intermediate depths across the
    very boundaries this loss is about). Gradients count only where both
    neighbours are valid -- zeroing alone would leave a spurious |0 - d| edge
    along every valid/invalid border.
    """
    log_target = torch.where(valid, target, torch.ones_like(target)).log()
    diff = torch.where(valid, torch.log(pred) - log_target, torch.zeros_like(pred))
    total = pred.new_zeros(())
    for k in range(GRAD_SCALES):
        step = 2**k
        d, v = diff[::step, ::step], valid[::step, ::step]
        grad_x = (d[:, 1:] - d[:, :-1]).abs() * (v[:, 1:] & v[:, :-1]).float()
        grad_y = (d[1:, :] - d[:-1, :]).abs() * (v[1:, :] & v[:-1, :]).float()
        total = total + (grad_x.sum() + grad_y.sum()) / v.sum().clamp_min(1)
    return total


def _grad_ratio(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Sharpness probe (eval only): mean |grad log pred| / |grad log gt| over
    valid pairs. Below 1 means blurrier than the pseudolabel; above 1 means
    hallucinated texture (a sign GRAD_WEIGHT is too high).
    """
    log_pred = torch.log(pred)
    log_target = torch.where(valid, target, torch.ones_like(target)).log()
    vx = valid[:, 1:] & valid[:, :-1]
    vy = valid[1:, :] & valid[:-1, :]
    pred_mag = (log_pred[:, 1:] - log_pred[:, :-1]).abs()[vx].sum() + (log_pred[1:, :] - log_pred[:-1, :]).abs()[vy].sum()
    gt_mag = (log_target[:, 1:] - log_target[:, :-1]).abs()[vx].sum() + (log_target[1:, :] - log_target[:-1, :]).abs()[vy].sum()
    return pred_mag / gt_mag.clamp_min(1e-8)


def loss(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Combined loss for one frame. pred and target are [H, W] metric depth.
    Returns the components too, so the caller can log them separately.
    """
    valid = _valid_mask(target)
    silog = _loss_silog(pred, target, valid)
    hn = _loss_hn(pred, target, valid)
    grad = _loss_grad(pred, target, valid)
    return {
        "silog": silog,
        "hn": hn,
        "grad": grad,
        "total": silog + HN_WEIGHT * hn + GRAD_WEIGHT * grad,
    }


# ----------------------------------------------------------------------------- model io

def predict(model: DepthDecoder, features: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """
    features: [1, C, H, W] V-JEPA grid -> metric depth [H_img, W_img] at
    pseudolabel resolution (DPT output is bilinearly upsampled to match).

    The DPT runs under bf16 autocast; the depth map is upcast to fp32 before the
    upsample so the SiLog/HN losses (log, median, MAD) stay numerically stable.
    """
    with autocast(DEVICE):
        depth = model(features)  # [1, 1, h, w]
    depth = F.interpolate(depth.float(), size=target_shape, mode="bilinear", align_corners=False)
    return depth[0, 0]


# ----------------------------------------------------------------------------- loops

@torch.no_grad()
def evaluate(model: DepthDecoder, loader) -> dict[str, float]:
    """Forward-only pass over `loader`; mean loss components plus AbsRel."""
    model.eval()
    totals = {"silog": 0.0, "hn": 0.0, "grad": 0.0, "total": 0.0, "absrel": 0.0, "grad_ratio": 0.0}
    n_frames = 0

    for batch in loader:
        features = batch["features"].to(DEVICE, non_blocking=True)  # [1, C, H, W]
        target = batch["target"][0].to(DEVICE, non_blocking=True)   # [H_img, W_img]

        pred = predict(model, features, target.shape)
        losses = loss(pred, target)

        valid = _valid_mask(target)
        absrel = ((pred[valid] - target[valid]).abs() / target[valid]).mean()

        for key in ("silog", "hn", "grad", "total"):
            totals[key] += losses[key].item()
        totals["absrel"] += absrel.item()
        totals["grad_ratio"] += _grad_ratio(pred, target, valid).item()
        n_frames += 1

    return {key: value / max(n_frames, 1) for key, value in totals.items()}


def save_checkpoint(
    model: DepthDecoder,
    iteration: int,
    metrics: dict[str, float],
    path: Path,
    bins_strategy: str,
    norm_strategy: str,
) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "validation_metrics": metrics,
            "config": {
                "min_depth": MIN_DEPTH,
                "max_depth": MAX_DEPTH,
                "silog_lambda": SILOG_LAMBDA,
                "hn_scales": HN_SCALES,
                "hn_weight": HN_WEIGHT,
                "grad_scales": GRAD_SCALES,
                "grad_weight": GRAD_WEIGHT,
                # The state dict is shape-identical across strategies, so this
                # config entry is the only record of which readout was trained.
                "bins_strategy": bins_strategy,
                "norm_strategy": norm_strategy,
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dense depth decoder")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--val-every", type=int, default=VAL_EVERY)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-frac", type=float, default=WARMUP_FRAC)
    parser.add_argument("--bins-strategy", choices=["linear", "log"], default="linear")
    parser.add_argument("--norm-strategy", choices=["linear", "softmax"], default="linear")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    loaders = KITTIDenseLoaders(
        task="depth",
        split=SPLIT_V1,
        batch_size=1,
        num_workers=args.num_workers,
    )
    print(loaders)

    total_opt_steps = args.max_iters // args.grad_accum
    wandb.init(
        project="jepa-drive-wm",
        job_type="depth_decoder",
        config={
            "max_iters": args.max_iters,
            "grad_accum": args.grad_accum,
            "effective_batch": args.grad_accum,
            "num_workers": args.num_workers,
            "amp_dtype": "bfloat16",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "min_depth": MIN_DEPTH,
            "max_depth": MAX_DEPTH,
            "silog_lambda": SILOG_LAMBDA,
            "hn_scales": HN_SCALES,
            "hn_weight": HN_WEIGHT,
            "hn_min_valid_per_tile": HN_MIN_VALID_PER_TILE,
            "grad_scales": GRAD_SCALES,
            "grad_weight": GRAD_WEIGHT,
            "bins_strategy": args.bins_strategy,
            "norm_strategy": args.norm_strategy,
            "train_sequences": TRAIN_SEQUENCES,
            "validation_sequences": VALIDATION_SEQUENCES,
            "test_sequences": TEST_SEQUENCES,
        },
    )
    wandb.define_metric("iter")
    wandb.define_metric("train/*", step_metric="iter")
    wandb.define_metric("val/*", step_metric="iter")

    model = DepthDecoder(bins_strategy=args.bins_strategy, norm_strategy=args.norm_strategy).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine(optimizer, total_opt_steps, int(args.warmup_frac * total_opt_steps))

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    train_batches = infinite(loaders.train)
    running = {"silog": 0.0, "hn": 0.0, "grad": 0.0, "total": 0.0}
    optimizer.zero_grad(set_to_none=True)
    model.train()
    start = time.time()

    for iteration in range(1, args.max_iters + 1):
        batch = next(train_batches)
        features = batch["features"].to(DEVICE, non_blocking=True)  # [1, C, H, W]
        target = batch["target"][0].to(DEVICE, non_blocking=True)   # [H_img, W_img]

        pred = predict(model, features, target.shape)
        losses = loss(pred, target)
        # Scale by 1/accum so accumulated grads average the micro-batches.
        (losses["total"] / args.grad_accum).backward()

        if iteration % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        for key in running:
            running[key] += losses[key].item()

        if iteration % args.log_every == 0:
            rate = iteration / (time.time() - start)
            means = {key: value / args.log_every for key, value in running.items()}
            running = {key: 0.0 for key in running}
            print(
                f"iter {iteration:6d}/{args.max_iters} | total {means['total']:.4f} "
                f"(silog {means['silog']:.4f}, hn {means['hn']:.4f}, grad {means['grad']:.4f}) | "
                f"lr {scheduler.get_last_lr()[0]:.2e} | {rate:.1f} it/s",
                flush=True,
            )
            wandb.log(
                {
                    "iter": iteration,
                    "train/total": means["total"],
                    "train/silog": means["silog"],
                    "train/hn": means["hn"],
                    "train/grad": means["grad"],
                    "train/lr": scheduler.get_last_lr()[0],
                }
            )

        if iteration % args.val_every == 0 or iteration == args.max_iters:
            validation_metrics = evaluate(model, loaders.validation)
            model.train()
            print(
                f"  [val] iter {iteration} | total {validation_metrics['total']:.4f} "
                f"absrel {validation_metrics['absrel']:.4f} "
                f"grad_ratio {validation_metrics['grad_ratio']:.4f}",
                flush=True,
            )
            wandb.log(
                {
                    "iter": iteration,
                    "val/total": validation_metrics["total"],
                    "val/silog": validation_metrics["silog"],
                    "val/hn": validation_metrics["hn"],
                    "val/grad": validation_metrics["grad"],
                    "val/absrel": validation_metrics["absrel"],
                    "val/grad_ratio": validation_metrics["grad_ratio"],
                }
            )
            if validation_metrics["total"] < best_validation_loss:
                best_validation_loss = validation_metrics["total"]
                save_checkpoint(model, iteration, validation_metrics, args.checkpoint,
                                args.bins_strategy, args.norm_strategy)
                wandb.summary["best_val_total"] = validation_metrics["total"]
                wandb.summary["best_val_absrel"] = validation_metrics["absrel"]
                wandb.summary["best_iter"] = iteration
                print(f"          -> saved new best model to {args.checkpoint}")

    # No test-split evaluation here: training owns train+validation only.
    # Test metrics come from evals/depth_evaluator.py on the saved checkpoint.
    wandb.finish()


if __name__ == "__main__":
    main()