"""
Training script for the VJEPA2.1 World Model.

Big idea:

- Get training windows of context latents, future latents and ego motions.
- Teacher forcing loss: predict each future frame from a ground-truth history.
- Rollout loss: predict the future autoregressively, feeding predictions back
  as context, with gradients flowing through the fed-back predictions.
- Total loss = teacher forcing + rollout.

Unlike VJEPA2-AC's normalize_reps=True training configuration, this model
operates directly in the frozen V-JEPA 2.1 encoder representation space Z.
Both teacher-forcing and rollout losses are L1 losses between predicted and
ground-truth V-JEPA latents, with no additional representation normalisation.
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from jaxtyping import Float
from torch import Tensor

from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.training_utils import autocast, build_warmup_cosine, infinite
from jepa_drive_wm.data.data_interface_rollout import KITTIRolloutLoaders
from jepa_drive_wm.data.splits import SPLIT_V1
from jepa_drive_wm.models.predictors.ac_style.ac_predictor import VJEPA21WorldModel

# Speed ceiling used to normalise per-step translation. Generous for road
# vehicles (~90 mph); KITTI's fastest highway stretches reach ~34 m/s.
MAX_SPEED_MPS = 40.0


def action_scale(step_seconds: float) -> torch.Tensor:
    """Per-step normalisation for raw ego motion (forward, right, yaw_right).

    The Fourier action embedder (frequencies 2^k * pi) is exactly periodic
    with period 2 in its input, so inputs outside one period alias onto each
    other. Dividing translation by (MAX_SPEED_MPS * step_seconds) guarantees
    |forward|, |right| <= 1 for any vehicle obeying the speed ceiling -- always
    within one period, never aliasing -- and typical steps land well inside
    +-0.5, where the coarsest band is monotonic.

    Yaw is divided by pi, mapping [-pi, pi] onto one full embedding period so
    the embedding's wraparound coincides with the physical wraparound: a yaw
    of -pi and +pi (the same rotation) get the same embedding.
    """
    max_translation = MAX_SPEED_MPS * step_seconds
    return torch.tensor([max_translation, max_translation, math.pi])


def to_grid(
    latents: Float[Tensor, "batch time hw latent"], H: int, W: int
) -> Float[Tensor, "batch time height width latent"]:
    """The dataset stores each frame as flat tokens; the model wants a grid."""
    B, T, N, C = latents.shape
    assert N == H * W, f"expected {H}x{W}={H * W} tokens, got {N}"
    return latents.view(B, T, H, W, C)


def latent_loss(
    pred: Float[Tensor, "batch steps height width latent"],
    target: Float[Tensor, "batch steps height width latent"],
) -> Float[Tensor, ""]:
    """L1 directly in the V-JEPA 2.1 latent space.

    Predictions may arrive in bf16 under autocast; compute the loss in fp32.
    """
    return F.l1_loss(pred.float(), target.float())


def forward_predictions(
    model: VJEPA21WorldModel,
    context: Float[Tensor, "batch time height width latent"],
    ego_motions: Float[Tensor, "batch steps 3"],
    future: Float[Tensor, "batch steps height width latent"],
) -> tuple[
    Float[Tensor, "batch steps height width latent"],
    Float[Tensor, "batch steps height width latent"],
]:
    """Readable equivalent of VJEPA2-AC's forward_predictions.

    Teacher forcing: every step predicts from a ground-truth history.
    Autoregressive: predictions are fed back as context (gradients flow
    through them). Both use a sliding window of the original context length.
    """
    T = context.shape[1]
    K = ego_motions.shape[1]

    def slide(window, next_frame):
        return torch.cat([window, next_frame.unsqueeze(1)], dim=1)[:, -T:]

    # -- teacher forcing
    tf_preds = []
    window = context
    for k in range(K):
        tf_preds.append(model(window, ego_motions[:, k]))
        window = slide(window, future[:, k])

    # -- autoregressive rollout (step 0 is the same prediction as TF step 0)
    ar_preds = [tf_preds[0]]
    window = slide(context, tf_preds[0])
    for k in range(1, K):
        pred = model(window, ego_motions[:, k])
        ar_preds.append(pred)
        window = slide(window, pred)

    return torch.stack(tf_preds, dim=1), torch.stack(ar_preds, dim=1)


def batch_losses(
    model: VJEPA21WorldModel,
    batch: dict,
    device: torch.device,
    act_scale: torch.Tensor,
) -> tuple[Tensor, Tensor]:
    """Teacher-forcing and rollout losses for one batch, under bf16 autocast.

    `act_scale` is action_scale(step_seconds) already on-device; passing it in
    keeps the action normalisation identical between train and eval.
    """
    context = to_grid(
        batch["context_latents"].to(device), model.grid_height, model.grid_width
    )
    future = to_grid(
        batch["future_latents"].to(device), model.grid_height, model.grid_width
    )
    ego_motions = batch["future_ego_motions"].to(device) / act_scale

    with autocast(device):
        tf_preds, ar_preds = forward_predictions(model, context, ego_motions, future)
    # latent_loss upcasts to fp32 internally.
    return latent_loss(tf_preds, future), latent_loss(ar_preds, future)


@torch.no_grad()
def evaluate(
    model: VJEPA21WorldModel,
    loader,
    device: torch.device,
    step_seconds: float,
) -> tuple[float, float]:
    """Forward-only pass; mean (teacher forcing, rollout) losses."""
    model.eval()
    act_scale = action_scale(step_seconds).to(device)
    tf_total, ar_total = 0.0, 0.0
    for batch in loader:
        tf_loss, ar_loss = batch_losses(model, batch, device, act_scale)
        tf_total += tf_loss.item()
        ar_total += ar_loss.item()
    n = max(len(loader), 1)
    return tf_total / n, ar_total / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the VJEPA2.1 world model")
    # Iteration budget (micro-steps = batches), not epochs. The world model has
    # no prior run to calibrate against, so this is a sensible first budget:
    # 40k batches of 2 over ~18.9k windows is ~4 effective passes. Watch val/ar
    # and extend with --max-iters if it is still improving.
    parser.add_argument("--max-iters", type=int, default=40_000)
    parser.add_argument("--grad-accum", type=int, default=4)  # eff. batch = grad_accum * batch_size
    parser.add_argument("--val-every", type=int, default=5_000)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)  # hide shared-FS read latency
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a few batches through train+val and exit, to prove the loop.",
    )
    parser.add_argument("--checkpoint", type=Path, default=OUTPUTS_DIR / "checkpoints_wm" / "world_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = KITTIRolloutLoaders(
        split=SPLIT_V1,
        context_length=args.context_length,
        future_length=args.future_length,
        frame_stride=args.frame_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(loaders)

    # `disabled` turns every wandb.log below into a no-op during smoke tests.
    wandb.init(
        project="jepa-drive-wm",
        job_type="world_model",
        config={
            **vars(args),
            "checkpoint": str(args.checkpoint),
            "max_speed_mps": MAX_SPEED_MPS,
            "step_seconds": loaders.step_seconds,
            "split": SPLIT_V1.name,
            "train_sequences": list(SPLIT_V1.train_sequences),
            "validation_sequences": list(SPLIT_V1.validation_sequences),
            "test_sequences": list(SPLIT_V1.test_sequences),
        },
        mode="disabled" if args.smoke_test else "online",
    )
    wandb.define_metric("iter")
    wandb.define_metric("train/*", step_metric="iter")
    wandb.define_metric("val/*", step_metric="iter")

    model = VJEPA21WorldModel().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    act_scale = action_scale(loaders.step_seconds).to(device)

    if args.smoke_test:
        from itertools import islice

        for batch in islice(loaders.train, 4):
            tf_loss, ar_loss = batch_losses(model, batch, device, act_scale)
            (tf_loss + ar_loss).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print("smoke test passed: data loads and a bf16 train step runs.")
        wandb.finish()
        return

    total_opt_steps = args.max_iters // args.grad_accum
    scheduler = build_warmup_cosine(optimizer, total_opt_steps, int(args.warmup_frac * total_opt_steps))

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val_ar = float("inf")

    train_batches = infinite(loaders.train)
    tf_running, ar_running = 0.0, 0.0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    start = time.time()

    for iteration in range(1, args.max_iters + 1):
        batch = next(train_batches)
        tf_loss, ar_loss = batch_losses(model, batch, device, act_scale)
        # Total loss = teacher forcing + rollout, scaled by 1/accum for averaging.
        ((tf_loss + ar_loss) / args.grad_accum).backward()

        grad_norm = None
        if iteration % args.grad_accum == 0:
            # clip_grad_norm_ returns the pre-clip norm: a free health metric.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        tf_running += tf_loss.item()
        ar_running += ar_loss.item()

        if iteration % args.log_every == 0:
            rate = iteration / (time.time() - start)
            log = {
                "iter": iteration,
                "train/tf": tf_running / args.log_every,
                "train/ar": ar_running / args.log_every,
                "train/lr": scheduler.get_last_lr()[0],
            }
            if grad_norm is not None:
                log["train/grad_norm"] = grad_norm.item()
            print(
                f"iter {iteration:6d}/{args.max_iters} | "
                f"tf {log['train/tf']:.4f} ar {log['train/ar']:.4f} | "
                f"lr {log['train/lr']:.2e} | {rate:.1f} it/s",
                flush=True,
            )
            wandb.log(log)
            tf_running, ar_running = 0.0, 0.0

        if iteration % args.val_every == 0 or iteration == args.max_iters:
            val_tf, val_ar = evaluate(model, loaders.validation, device, loaders.step_seconds)
            model.train()
            print(f"  [val] iter {iteration} | tf {val_tf:.4f} ar {val_ar:.4f}", flush=True)
            wandb.log({"iter": iteration, "val/tf": val_tf, "val/ar": val_ar})

            # The rollout loss is the one that matters for navigation, so it
            # selects the checkpoint.
            if val_ar < best_val_ar:
                best_val_ar = val_ar
                torch.save(
                    {
                        "model": model.state_dict(),
                        "iteration": iteration,
                        "val_ar": val_ar,
                        # The action normalisation is part of the model contract:
                        # inference must divide raw ego motion by the same scale.
                        "max_speed_mps": MAX_SPEED_MPS,
                        "step_seconds": loaders.step_seconds,
                    },
                    args.checkpoint,
                )
                wandb.summary["best_val_ar"] = val_ar
                wandb.summary["best_iter"] = iteration
                print(f"          saved checkpoint (val ar {val_ar:.4f})")

    wandb.finish()


if __name__ == "__main__":
    main()