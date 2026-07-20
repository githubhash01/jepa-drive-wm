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
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

from jepa_drive_wm.dense_decoder.data_interface_dense import KITTIDenseLoaders
from jepa_drive_wm.dense_decoder.model_semantics import SemanticDecoder

# ----------------------------------------------------------------------------- config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 19       # Cityscapes train ids, as produced by OneFormer
IGNORE_INDEX = 255     # pixels OneFormer left unlabeled

EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

TRAIN_SEQUENCES = [0, 1, 2, 3, 5, 6, 8]
VALIDATION_SEQUENCES = [7, 10]
TEST_SEQUENCES = [4, 9]

CHECKPOINT_PATH = Path("/home/hashim/Desktop/jepa-drive-wm/src/jepa_drive_wm/dense_decoder/checkpoints_semantics/semantic_decoder_best.pt")


# ----------------------------------------------------------------------------- model io

def predict(model: SemanticDecoder, features: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """
    features: [1, C, H, W] V-JEPA grid -> class logits [1, 19, H_img, W_img]
    at pseudolabel resolution (DPT output is bilinearly upsampled to match).
    """
    logits = model(features)  # [1, 19, h, w]
    return F.interpolate(logits, size=target_shape, mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------- loops

def run_epoch(
    model: SemanticDecoder,
    loader,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """
    One pass over `loader`. Trains if an optimizer is given, else evaluates.
    Returns mean cross entropy and pixel accuracy.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_accuracy = 0.0
    n_frames = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            features = batch["features"].to(DEVICE, non_blocking=True)  # [1, C, H, W]
            target = batch["target"].to(DEVICE, non_blocking=True)      # [1, H_img, W_img]

            logits = predict(model, features, target.shape[-2:])
            loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                labeled = target != IGNORE_INDEX
                correct = logits.argmax(dim=1)[labeled] == target[labeled]
                accuracy = correct.float().mean() if labeled.any() else torch.zeros(())

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            n_frames += 1

    n = max(n_frames, 1)
    return {"loss": total_loss / n, "accuracy": total_accuracy / n}


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

    wandb.init(
        project="jepa-drive-wm",
        job_type="semantic_decoder",
        config={
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "train_sequences": TRAIN_SEQUENCES,
            "validation_sequences": VALIDATION_SEQUENCES,
            "test_sequences": TEST_SEQUENCES,
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    model = SemanticDecoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, loaders.train, optimizer)
        validation_metrics = run_epoch(model, loaders.validation)

        print(
            f"epoch {epoch:3d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f} acc {validation_metrics['accuracy']:.4f}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                # Logged before scheduler.step(): the LR these batches saw.
                "train/lr": scheduler.get_last_lr()[0],
                "val/loss": validation_metrics["loss"],
                "val/accuracy": validation_metrics["accuracy"],
            }
        )
        scheduler.step()

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                    "config": {
                        "num_classes": NUM_CLASSES,
                        "ignore_index": IGNORE_INDEX,
                    },
                },
                CHECKPOINT_PATH,
            )
            wandb.summary["best_val_loss"] = validation_metrics["loss"]
            wandb.summary["best_val_accuracy"] = validation_metrics["accuracy"]
            wandb.summary["best_epoch"] = epoch
            print(f"          -> saved new best model to {CHECKPOINT_PATH}")

    test_metrics = run_epoch(model, loaders.test)
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