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
from jaxtyping import Float
from torch import Tensor

from jepa_drive_wm.world_model.data_interface_wm import KITTIRolloutLoaders
from jepa_drive_wm.world_model.model import VJEPA21WorldModel

# TODO - understand wtf is going on here
# The Fourier action embedder expects inputs in roughly [-0.5, 0.5].
# At 0.5s steps, KITTI translation per step is at most ~17m (highway); yaw is
# wrapped to [-pi, pi]. These constants map raw (dx, dy, yaw) into range.
ACTION_SCALE = torch.tensor([20.0, 20.0, math.pi])


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
    optimizer: torch.optim.Optimizer | None = None,
    log_every: int = 50,
    tag: str = "train",
) -> tuple[float, float]:
    """One pass over a loader. Trains if an optimizer is given, else evaluates.
    Returns mean (teacher forcing, rollout) losses.

    Logs the moment the first batch arrives (so a stalled data pipeline is
    obvious immediately) and a running average every `log_every` steps.
    """
    training = optimizer is not None
    model.train(training)
    action_scale = ACTION_SCALE.to(device)

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
        ego_motions = batch["future_ego_motions"].to(device) / action_scale

        with torch.set_grad_enabled(training):
            tf_preds, ar_preds = forward_predictions(model, context, ego_motions, future)
            tf_loss = latent_loss(tf_preds, future)
            ar_loss = latent_loss(ar_preds, future)
            loss = tf_loss + ar_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
    return tf_total / n, ar_total / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the VJEPA2.1 world model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
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
    parser.add_argument("--checkpoint", type=Path, default=Path("world_model.pt"))
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

    model = VJEPA21WorldModel().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.smoke_test:
        from itertools import islice

        few = list(islice(loaders.train, 4))
        run_epoch(model, few, device, optimizer, log_every=1, tag="smoke")
        print("smoke test passed: data loads and a train step runs.")
        return

    best_val_ar = float("inf")
    for epoch in range(args.epochs):
        train_tf, train_ar = run_epoch(
            model, loaders.train, device, optimizer, log_every=args.log_every, tag="train"
        )
        val_tf, val_ar = run_epoch(model, loaders.validation, device, log_every=0, tag="val")

        print(
            f"epoch {epoch:3d} | "
            f"train tf {train_tf:.4f} ar {train_ar:.4f} | "
            f"val tf {val_tf:.4f} ar {val_ar:.4f}"
        )

        # The rollout loss is the one that matters for navigation, so it
        # selects the checkpoint.
        if val_ar < best_val_ar:
            best_val_ar = val_ar
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_ar": val_ar},
                args.checkpoint,
            )
            print(f"          saved checkpoint (val ar {val_ar:.4f})")


if __name__ == "__main__":
    main()