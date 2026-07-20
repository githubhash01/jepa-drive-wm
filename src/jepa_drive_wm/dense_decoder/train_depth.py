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

The best model (lowest validation loss) is saved to CHECKPOINT_PATH.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

from jepa_drive_wm.dense_decoder.data_interface_dense import KITTIDenseLoaders
from jepa_drive_wm.dense_decoder.model_depth import DepthDecoder

# ----------------------------------------------------------------------------- config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Depth validity range; must match FeaturesToDepth(min_depth=0.5, max_depth=80.0).
MIN_DEPTH = 0.5
MAX_DEPTH = 80.0

SILOG_LAMBDA = 0.85          # balancing parameter from the paper
HN_SCALES = (1, 4, 8)        # M: image tiled into an M x M grid per scale
HN_WEIGHT = 1.0              # total = silog + HN_WEIGHT * hn
HN_MIN_VALID_PER_TILE = 32   # skip tiles with fewer valid pixels (stats too noisy)

EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

TRAIN_SEQUENCES = [0, 1, 2, 3, 5, 6, 8]
VALIDATION_SEQUENCES = [7, 10]
TEST_SEQUENCES = [4, 9]

CHECKPOINT_PATH = Path("/home/hashim/Desktop/jepa-drive-wm/src/jepa_drive_wm/dense_decoder/checkpoints_depth/depth_decoder_best.pt")


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


def loss(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Combined loss for one frame. pred and target are [H, W] metric depth.
    Returns the components too, so run_epoch can log them separately.
    """
    valid = _valid_mask(target)
    silog = _loss_silog(pred, target, valid)
    hn = _loss_hn(pred, target, valid)
    return {
        "silog": silog,
        "hn": hn,
        "total": silog + HN_WEIGHT * hn,
    }


# ----------------------------------------------------------------------------- model io

def predict(model: DepthDecoder, features: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """
    features: [1, C, H, W] V-JEPA grid -> metric depth [H_img, W_img] at
    pseudolabel resolution (DPT output is bilinearly upsampled to match).
    """
    depth = model(features)  # [1, 1, h, w]
    depth = F.interpolate(depth, size=target_shape, mode="bilinear", align_corners=False)
    return depth[0, 0]


# ----------------------------------------------------------------------------- loops

def run_epoch(
    model: DepthDecoder,
    loader,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """
    One pass over `loader`. Trains if an optimizer is given, else evaluates.
    Returns mean loss components plus AbsRel as a sanity metric.
    """
    training = optimizer is not None
    model.train(training)

    totals = {"silog": 0.0, "hn": 0.0, "total": 0.0, "absrel": 0.0}
    n_frames = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            features = batch["features"].to(DEVICE, non_blocking=True)  # [1, C, H, W]
            target = batch["target"][0].to(DEVICE, non_blocking=True)   # [H_img, W_img]

            pred = predict(model, features, target.shape)
            losses = loss(pred, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()

            with torch.no_grad():
                valid = _valid_mask(target)
                absrel = ((pred[valid] - target[valid]).abs() / target[valid]).mean()

            for key in ("silog", "hn", "total"):
                totals[key] += losses[key].item()
            totals["absrel"] += absrel.item()
            n_frames += 1

    return {key: value / max(n_frames, 1) for key, value in totals.items()}


def main() -> None:
    loaders = KITTIDenseLoaders(
        task="depth",
        training_sequences=TRAIN_SEQUENCES,
        validation_sequences=VALIDATION_SEQUENCES,
        test_sequences=TEST_SEQUENCES,
        batch_size=1,
        num_workers=4,
    )
    print(loaders)

    wandb.init(
        project="jepa-drive-wm",
        job_type="depth_decoder",
        config={
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "min_depth": MIN_DEPTH,
            "max_depth": MAX_DEPTH,
            "silog_lambda": SILOG_LAMBDA,
            "hn_scales": HN_SCALES,
            "hn_weight": HN_WEIGHT,
            "hn_min_valid_per_tile": HN_MIN_VALID_PER_TILE,
            "train_sequences": TRAIN_SEQUENCES,
            "validation_sequences": VALIDATION_SEQUENCES,
            "test_sequences": TEST_SEQUENCES,
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    model = DepthDecoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, loaders.train, optimizer)
        validation_metrics = run_epoch(model, loaders.validation)

        print(
            f"epoch {epoch:3d} | "
            f"train total {train_metrics['total']:.4f} "
            f"(silog {train_metrics['silog']:.4f}, hn {train_metrics['hn']:.4f}) | "
            f"val total {validation_metrics['total']:.4f} "
            f"absrel {validation_metrics['absrel']:.4f}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train/total": train_metrics["total"],
                "train/silog": train_metrics["silog"],
                "train/hn": train_metrics["hn"],
                "train/absrel": train_metrics["absrel"],
                # Logged before scheduler.step(): the LR these batches saw.
                "train/lr": scheduler.get_last_lr()[0],
                "val/total": validation_metrics["total"],
                "val/silog": validation_metrics["silog"],
                "val/hn": validation_metrics["hn"],
                "val/absrel": validation_metrics["absrel"],
            }
        )
        scheduler.step()

        if validation_metrics["total"] < best_validation_loss:
            best_validation_loss = validation_metrics["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                    "config": {
                        "min_depth": MIN_DEPTH,
                        "max_depth": MAX_DEPTH,
                        "silog_lambda": SILOG_LAMBDA,
                        "hn_scales": HN_SCALES,
                        "hn_weight": HN_WEIGHT,
                    },
                },
                CHECKPOINT_PATH,
            )
            wandb.summary["best_val_total"] = validation_metrics["total"]
            wandb.summary["best_val_absrel"] = validation_metrics["absrel"]
            wandb.summary["best_epoch"] = epoch
            print(f"          -> saved new best model to {CHECKPOINT_PATH}")

    test_metrics = run_epoch(model, loaders.test)
    print(f"test | total {test_metrics['total']:.4f} absrel {test_metrics['absrel']:.4f}")
    # A single point, not a curve: it belongs in the summary, not the timeline.
    wandb.summary.update(
        {
            "test/total": test_metrics["total"],
            "test/silog": test_metrics["silog"],
            "test/hn": test_metrics["hn"],
            "test/absrel": test_metrics["absrel"],
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()