"""
Training script for the VJEPA2.1 World Model.

Big idea:

- Get training windows of context latents, future latents and ego motions.
- Teacher forcing loss: predict each future frame from a ground-truth history.
- Rollout loss: predict the future autoregressively, feeding predictions back
  as context, with gradients flowing through the fed-back predictions.
- Total loss = teacher forcing + rollout (equal weight, as in VJEPA2-AC).

Following VJEPA2-AC, the loss is L1 between layer-normalised latents. The
first future step has no history to differ on, so it is shared between the
two losses (also as in VJEPA2-AC, where z_ar starts from z_tf's first frame).
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
from jepa_drive_wm.world_model.data_interface_wm import KITTIRolloutLoaders
from jepa_drive_wm.world_model.model import VJEPA21WorldModel

# Speed ceiling used to normalise per-step translation. Generous for road
# vehicles (~90 mph); KITTI's fastest highway stretches reach ~34 m/s.
MAX_SPEED_MPS = 40.0


def action_scale(step_seconds: float) -> torch.Tensor:
    """Per-step normalisation for raw ego motion (dx, dy, yaw).

    The Fourier action embedder (frequencies 2^k * pi) is exactly periodic
    with period 2 in its input, so inputs outside one period alias onto each
    other. Dividing translation by (MAX_SPEED_MPS * step_seconds) guarantees
    |dx|, |dy| <= 1 for any vehicle obeying the speed ceiling -- always within
    one period, never aliasing -- and typical steps land well inside +-0.5,
    where the coarsest band is monotonic.

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
    """L1 between layer-normalised latents (VJEPA2-AC's normalize_reps)."""
    pred = F.layer_norm(pred, (pred.size(-1),))
    target = F.layer_norm(target, (target.size(-1),))
    return F.l1_loss(pred, target)


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


def run_epoch(
    model: VJEPA21WorldModel,
    loader,
    device: torch.device,
    step_seconds: float,
    optimizer: torch.optim.Optimizer | None = None,
    log_every: int = 50,
    tag: str = "train",
    global_step: int = 0,
) -> tuple[float, float, int]:
    """One pass over a loader. Trains if an optimizer is given, else evaluates.
    Returns mean (teacher forcing, rollout) losses and the updated global step.

    `step_seconds` is the physical duration of one prediction step; take it
    from KITTIRolloutLoaders.step_seconds rather than recomputing it, so the
    action normalisation always agrees with the data pipeline. Passing it
    explicitly also keeps this function working on any iterable of batches
    (e.g. the smoke test's plain list), not just a DataLoader.

    Logs the moment the first batch arrives (so a stalled data pipeline is
    obvious immediately) and a running average every `log_every` steps.
    Per-batch wandb logging happens only when training; per-epoch logging is
    the caller's job.
    """
    training = optimizer is not None
    model.train(training)
    act_scale = action_scale(step_seconds).to(device)

    tf_total, ar_total = 0.0, 0.0
    start = time.time()
    for step, batch in enumerate(loader):
        if step == 0:
            print(f"  [{tag}] first batch in {time.time() - start:.1f}s", flush=True)

        context = to_grid(
            batch["context_latents"].to(device), model.grid_height, model.grid_width
        )
        future = to_grid(
            batch["future_latents"].to(device), model.grid_height, model.grid_width
        )
        ego_motions = batch["future_ego_motions"].to(device) / act_scale

        with torch.set_grad_enabled(training):
            tf_preds, ar_preds = forward_predictions(model, context, ego_motions, future)
            tf_loss = latent_loss(tf_preds, future)
            ar_loss = latent_loss(ar_preds, future)
            loss = tf_loss + ar_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # clip_grad_norm_ returns the pre-clip norm: a free health metric.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            wandb.log(
                {
                    "batch/step": global_step,
                    "batch/tf": tf_loss.item(),
                    "batch/ar": ar_loss.item(),
                    "batch/grad_norm": grad_norm.item(),
                }
            )
            global_step += 1

        tf_total += tf_loss.item()
        ar_total += ar_loss.item()

        if log_every and (step + 1) % log_every == 0:
            rate = (step + 1) / (time.time() - start)
            print(
                f"  [{tag}] step {step + 1}/{len(loader)} | "
                f"tf {tf_total / (step + 1):.4f} ar {ar_total / (step + 1):.4f} | "
                f"{rate:.1f} it/s",
                flush=True,
            )

    n = max(len(loader), 1)
    return tf_total / n, ar_total / n, global_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the VJEPA2.1 world model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a few batches through train+val and exit, to prove the loop.",
    )
    parser.add_argument("--checkpoint", type=Path, default=OUTPUTS_DIR / "checkpoints_wm" / "world_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = KITTIRolloutLoaders(
        training_sequences=[0, 1, 2, 3, 5, 6, 8],
        validation_sequences=[7, 10],
        test_sequences=[4, 9],
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
        },
        mode="disabled" if args.smoke_test else "online",
    )
    # Epoch metrics plot against `epoch`; batch metrics against their own
    # counter. Without this, per-batch and per-epoch logs fight over wandb's
    # global step and the epoch curves come out mangled.
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("batch/step")
    wandb.define_metric("batch/*", step_metric="batch/step")

    model = VJEPA21WorldModel().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.smoke_test:
        from itertools import islice

        few = list(islice(loaders.train, 4))
        run_epoch(
            model, few, device, loaders.step_seconds, optimizer, log_every=1, tag="smoke"
        )
        print("smoke test passed: data loads and a train step runs.")
        wandb.finish()
        return

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_val_ar = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        train_tf, train_ar, global_step = run_epoch(
            model,
            loaders.train,
            device,
            loaders.step_seconds,
            optimizer,
            log_every=args.log_every,
            tag="train",
            global_step=global_step,
        )
        val_tf, val_ar, _ = run_epoch(
            model, loaders.validation, device, loaders.step_seconds, log_every=0, tag="val"
        )

        print(
            f"epoch {epoch:3d} | "
            f"train tf {train_tf:.4f} ar {train_ar:.4f} | "
            f"val tf {val_tf:.4f} ar {val_ar:.4f}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train/tf": train_tf,
                "train/ar": train_ar,
                "val/tf": val_tf,
                "val/ar": val_ar,
            }
        )

        # The rollout loss is the one that matters for navigation, so it
        # selects the checkpoint.
        if val_ar < best_val_ar:
            best_val_ar = val_ar
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_ar": val_ar,
                    # The action normalisation is part of the model contract:
                    # inference must divide raw ego motion by the same scale.
                    "max_speed_mps": MAX_SPEED_MPS,
                    "step_seconds": loaders.step_seconds,
                },
                args.checkpoint,
            )
            wandb.summary["best_val_ar"] = val_ar
            wandb.summary["best_epoch"] = epoch
            print(f"          saved checkpoint (val ar {val_ar:.4f})")

    wandb.finish()


if __name__ == "__main__":
    main()