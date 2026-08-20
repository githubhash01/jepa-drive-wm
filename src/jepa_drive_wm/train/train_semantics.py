"""
Train the dense semantic decoder (SemanticDecoder) on KITTI OneFormer pseudolabels.

Pipeline per sample (batch_size=1):

    V-JEPA grid [1, C, H, W]
        -> SemanticDecoder (4x repeated final features -> DINOv3 DPT -> 19 logits)
        -> logits [1, 19, h, w]
        -> bilinear upsample to the pseudolabel resolution
        -> cross entropy against OneFormer class ids [1, H_img, W_img]

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
from jepa_drive_wm.models.dense_decoders.semantic_decoder import SemanticDecoder
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.training_utils import autocast, build_warmup_cosine, infinite, init_run

# ----------------------------------------------------------------------------- config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 19       # Cityscapes train ids, as produced by OneFormer
IGNORE_INDEX = 255     # pixels OneFormer left unlabeled

# Iteration budget, not epochs (see training_utils / train_depth for the rationale).
MAX_ITERS = 50_000
GRAD_ACCUM = 8               # effective batch size = GRAD_ACCUM * (loader batch 1)
VAL_EVERY = 5_000            # micro-steps between validation + checkpoint
LOG_EVERY = 100
NUM_WORKERS = 8              # dedicated training box; more workers hide shared-FS read latency
WARMUP_FRAC = 0.03           # fraction of optimizer steps spent warming the LR up
LEARNING_RATE = 3e-4         # peak LR (matches the old dpt probe runs' peak_lr)
WEIGHT_DECAY = 0.01

# All of the above are defaults; every one is overridable on the command line
# (see main's argparse) so budgets can be tuned per-machine without editing files.

# Sequence assignment lives in data/splits.py; init_run records the full split in
# the wandb config. Test metrics come from evals/semantics_evaluator.py, which
# reads the split itself.

CHECKPOINT_PATH = OUTPUTS_DIR / "checkpoints_semantics" / "semantic_decoder_best.pt"


# ----------------------------------------------------------------------------- model io

def predict(model: SemanticDecoder, features: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """
    features: [1, C, H, W] V-JEPA grid -> class logits [1, 19, H_img, W_img]
    at pseudolabel resolution (DPT output is bilinearly upsampled to match).

    The DPT runs under bf16 autocast; logits are upcast to fp32 before the
    upsample so the cross-entropy is computed in full precision.
    """
    with autocast(DEVICE):
        logits = model(features)  # [1, 19, h, w]
    return F.interpolate(logits.float(), size=target_shape, mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------- loops

@torch.no_grad()
def evaluate(model: SemanticDecoder, loader) -> dict[str, float]:
    """Forward-only pass over `loader`; mean cross entropy and pixel accuracy."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    n_frames = 0

    for batch in loader:
        features = batch["features"].to(DEVICE, non_blocking=True)  # [1, C, H, W]
        target = batch["target"].to(DEVICE, non_blocking=True)      # [1, H_img, W_img]

        logits = predict(model, features, target.shape[-2:])
        loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)

        labeled = target != IGNORE_INDEX
        correct = logits.argmax(dim=1)[labeled] == target[labeled]
        accuracy = correct.float().mean() if labeled.any() else torch.zeros(())

        total_loss += loss.item()
        total_accuracy += accuracy.item()
        n_frames += 1

    n = max(n_frames, 1)
    return {"loss": total_loss / n, "accuracy": total_accuracy / n}


def save_checkpoint(model: SemanticDecoder, iteration: int, metrics: dict[str, float], path: Path) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "validation_metrics": metrics,
            "config": {"num_classes": NUM_CLASSES, "ignore_index": IGNORE_INDEX},
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dense semantic decoder")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--val-every", type=int, default=VAL_EVERY)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-frac", type=float, default=WARMUP_FRAC)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    loaders = KITTIDenseLoaders(
        task="semantics",
        split=SPLIT_V1,
        batch_size=1,
        num_workers=args.num_workers,
    )
    print(loaders)

    total_opt_steps = args.max_iters // args.grad_accum

    model = SemanticDecoder().to(DEVICE)

    init_run(
        job_type="semantic_decoder",
        name=f"semantics-lr{args.lr:g}-{args.max_iters // 1000}k",
        tags=["semantics", "dense-decoder", "dpt"],
        config={
            **vars(args),
            "checkpoint": str(args.checkpoint),
            "effective_batch": args.grad_accum,  # loader batch is fixed at 1
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
        },
        split=SPLIT_V1,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine(optimizer, total_opt_steps, int(args.warmup_frac * total_opt_steps))

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    train_batches = infinite(loaders.train)
    running_loss = 0.0
    running_acc = 0.0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    start = time.time()

    for iteration in range(1, args.max_iters + 1):
        batch = next(train_batches)
        features = batch["features"].to(DEVICE, non_blocking=True)  # [1, C, H, W]
        target = batch["target"].to(DEVICE, non_blocking=True)      # [1, H_img, W_img]

        logits = predict(model, features, target.shape[-2:])
        loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
        (loss / args.grad_accum).backward()

        if iteration % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        with torch.no_grad():
            labeled = target != IGNORE_INDEX
            acc = (logits.argmax(dim=1)[labeled] == target[labeled]).float().mean() if labeled.any() else torch.zeros(())
        running_loss += loss.item()
        running_acc += acc.item()

        if iteration % args.log_every == 0:
            rate = iteration / (time.time() - start)
            mean_loss = running_loss / args.log_every
            mean_acc = running_acc / args.log_every
            running_loss = running_acc = 0.0
            print(
                f"iter {iteration:6d}/{args.max_iters} | loss {mean_loss:.4f} acc {mean_acc:.4f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e} | {rate:.1f} it/s",
                flush=True,
            )
            wandb.log(
                {
                    "iter": iteration,
                    "train/loss": mean_loss,
                    "train/accuracy": mean_acc,
                    "train/lr": scheduler.get_last_lr()[0],
                }
            )

        if iteration % args.val_every == 0 or iteration == args.max_iters:
            validation_metrics = evaluate(model, loaders.validation)
            model.train()
            print(
                f"  [val] iter {iteration} | loss {validation_metrics['loss']:.4f} "
                f"acc {validation_metrics['accuracy']:.4f}",
                flush=True,
            )
            wandb.log(
                {
                    "iter": iteration,
                    "val/loss": validation_metrics["loss"],
                    "val/accuracy": validation_metrics["accuracy"],
                }
            )
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                save_checkpoint(model, iteration, validation_metrics, args.checkpoint)
                wandb.summary["best_val_loss"] = validation_metrics["loss"]
                wandb.summary["best_val_accuracy"] = validation_metrics["accuracy"]
                wandb.summary["best_iter"] = iteration
                print(f"          -> saved new best model to {args.checkpoint}")

    # No test-split evaluation here: training owns train+validation only.
    # Test metrics come from evals/semantics_evaluator.py on the saved checkpoint.
    wandb.finish()


if __name__ == "__main__":
    main()