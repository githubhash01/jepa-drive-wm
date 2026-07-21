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
import time

import torch
import torch.nn.functional as F
import wandb

from jepa_drive_wm.dense_decoder.data_interface_dense import KITTIDenseLoaders
from jepa_drive_wm.dense_decoder.model_semantics import SemanticDecoder
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.training_utils import autocast, build_warmup_cosine, infinite

# ----------------------------------------------------------------------------- config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 19       # Cityscapes train ids, as produced by OneFormer
IGNORE_INDEX = 255     # pixels OneFormer left unlabeled

# Iteration budget, not epochs (see training_utils / train_depth for the rationale).
MAX_ITERS = 50_000
GRAD_ACCUM = 8               # effective batch size = GRAD_ACCUM * (loader batch 1)
VAL_EVERY = 5_000            # micro-steps between validation + checkpoint
LOG_EVERY = 100
WARMUP_FRAC = 0.03           # fraction of optimizer steps spent warming the LR up
LEARNING_RATE = 3e-4         # peak LR (matches the old dpt probe runs' peak_lr)
WEIGHT_DECAY = 0.01

# The dense decoders are per-frame and pose-free, so they train on ALL sequences
# with pseudolabels -- including 11-21, which have no GT poses and so are unusable
# for the world model but perfectly good here (this matches the old dpt recipe).
TRAIN_SEQUENCES = [0, 1, 2, 3, 5, 6, 8] + list(range(11, 22))
VALIDATION_SEQUENCES = [7, 10]
TEST_SEQUENCES = [4, 9]

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


def save_checkpoint(model: SemanticDecoder, iteration: int, metrics: dict[str, float]) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "validation_metrics": metrics,
            "config": {"num_classes": NUM_CLASSES, "ignore_index": IGNORE_INDEX},
        },
        CHECKPOINT_PATH,
    )


def main() -> None:
    loaders = KITTIDenseLoaders(
        task="semantics",
        training_sequences=TRAIN_SEQUENCES,
        validation_sequences=VALIDATION_SEQUENCES,
        test_sequences=TEST_SEQUENCES,
        batch_size=1,
        num_workers=4,
    )
    print(loaders)

    total_opt_steps = MAX_ITERS // GRAD_ACCUM
    wandb.init(
        project="jepa-drive-wm",
        job_type="semantic_decoder",
        config={
            "max_iters": MAX_ITERS,
            "grad_accum": GRAD_ACCUM,
            "effective_batch": GRAD_ACCUM,
            "amp_dtype": "bfloat16",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "train_sequences": TRAIN_SEQUENCES,
            "validation_sequences": VALIDATION_SEQUENCES,
            "test_sequences": TEST_SEQUENCES,
        },
    )
    wandb.define_metric("iter")
    wandb.define_metric("train/*", step_metric="iter")
    wandb.define_metric("val/*", step_metric="iter")

    model = SemanticDecoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = build_warmup_cosine(optimizer, total_opt_steps, int(WARMUP_FRAC * total_opt_steps))

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    train_batches = infinite(loaders.train)
    running_loss = 0.0
    running_acc = 0.0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    start = time.time()

    for iteration in range(1, MAX_ITERS + 1):
        batch = next(train_batches)
        features = batch["features"].to(DEVICE, non_blocking=True)  # [1, C, H, W]
        target = batch["target"].to(DEVICE, non_blocking=True)      # [1, H_img, W_img]

        logits = predict(model, features, target.shape[-2:])
        loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
        (loss / GRAD_ACCUM).backward()

        if iteration % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        with torch.no_grad():
            labeled = target != IGNORE_INDEX
            acc = (logits.argmax(dim=1)[labeled] == target[labeled]).float().mean() if labeled.any() else torch.zeros(())
        running_loss += loss.item()
        running_acc += acc.item()

        if iteration % LOG_EVERY == 0:
            rate = iteration / (time.time() - start)
            mean_loss = running_loss / LOG_EVERY
            mean_acc = running_acc / LOG_EVERY
            running_loss = running_acc = 0.0
            print(
                f"iter {iteration:6d}/{MAX_ITERS} | loss {mean_loss:.4f} acc {mean_acc:.4f} | "
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

        if iteration % VAL_EVERY == 0 or iteration == MAX_ITERS:
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
                save_checkpoint(model, iteration, validation_metrics)
                wandb.summary["best_val_loss"] = validation_metrics["loss"]
                wandb.summary["best_val_accuracy"] = validation_metrics["accuracy"]
                wandb.summary["best_iter"] = iteration
                print(f"          -> saved new best model to {CHECKPOINT_PATH}")

    test_metrics = evaluate(model, loaders.test)
    print(f"test | loss {test_metrics['loss']:.4f} acc {test_metrics['accuracy']:.4f}")
    # A single point, not a curve: it belongs in the summary, not the timeline.
    wandb.summary.update(
        {
            "test/loss": test_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()