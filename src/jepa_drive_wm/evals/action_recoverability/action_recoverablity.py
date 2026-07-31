"""
Probe how much ego motion is recoverable from frozen, cached V-JEPA 2.1 features.

An inverse dynamics model (TransformerIDM, idm.py) sees k+1 latent frames
spanning k action steps and regresses the k ego-motion actions
[forward (m), left (m), yaw (rad)] between them. If the frozen features carry
the geometry, a small probe should recover the actions; if they don't, no
world model built on them can be action-conditional in a meaningful way.

Recipe
- Same window machinery and splits as the world model: KITTIRolloutLoaders
  with context_length=1, future_length=k, frame_stride=5 (0.5 s steps).
- Step 1, training-set action statistics: per-component mean/std/median over
  every stride-step of the training sequences. Standardising with mean/std
  keeps the dominant forward translation from swamping left/yaw in the loss.
- Step 2, train the IDM (patch_size=3 over the 24x78 token grid to keep the
  parameter count down) with Smooth-L1 in standardised action space; best
  model by validation loss. Intended horizons: k=1 and k=4 (--horizon).

Reported per component, on the test sequences:
    MAE (metres for forward/left, degrees for yaw), R^2 (primary),
    Pearson r (secondary). Each baseline predicts a training-set constant --
    the one optimal for its metric: the median for MAE, the mean for R^2.
    Success = beating both.

Run:
    python -m jepa_drive_wm.evals.action_recoverability.action_recoverablity --horizon 1
    python -m jepa_drive_wm.evals.action_recoverability.action_recoverablity --horizon 4
    ... --horizon 1 --eval-only   # re-report from the saved checkpoint
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from jepa_drive_wm.data.kitti import VJEPA_BASE_EMBED_DIM, VJEPA_BASE_GRID_HW
from jepa_drive_wm.evals.action_recoverability.idm import create_idm
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.training_utils import autocast, build_warmup_cosine, infinite
from jepa_drive_wm.world_model.deterministic.data_interface_wm import (
    KITTIRolloutDataset,
    KITTIRolloutLoaders,
)

# ----------------------------------------------------------------------------- config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same splits as the world model; the probe is only meaningful on its data.
TRAIN_SEQUENCES = [0, 1, 2, 3, 5, 6, 8]
VALIDATION_SEQUENCES = [7, 10]
TEST_SEQUENCES = [4, 9]

GRID_H, GRID_W = VJEPA_BASE_GRID_HW

COMPONENT_NAMES = ("forward", "left", "yaw")
DISPLAY_UNIT = {"forward": "m", "left": "m", "yaw": "deg"}
DISPLAY_SCALE = {"forward": 1.0, "left": 1.0, "yaw": math.degrees(1.0)}


# ----------------------------------------------------------------------------- step 1: statistics

def action_statistics(dataset: KITTIRolloutDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-component mean/std/median of one-step ego motion, over every stride-step
    of the dataset's sequences -- exactly the support of the training windows'
    actions. Mean/std standardise the regression; median and mean are the
    constant baselines for MAE and R^2 respectively.
    """
    stride = dataset.frame_stride
    actions = np.array(
        [
            sequence.get_ego_motion(frame, frame + stride)
            for sequence in dataset.sequences.values()
            for frame in range(len(sequence) - stride)
        ],
        dtype=np.float64,
    )
    return (
        actions.mean(axis=0).astype(np.float32),
        actions.std(axis=0).astype(np.float32),
        np.median(actions, axis=0).astype(np.float32),
    )


# ----------------------------------------------------------------------------- model io

def chunk_grid(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """[B, 1, N, C] context + [B, k, N, C] future latents -> [B, k+1, H, W, C] IDM input."""
    latents = torch.cat([batch["context_latents"], batch["future_latents"]], dim=1)
    b, t, n, c = latents.shape
    return latents.view(b, t, GRID_H, GRID_W, c)


@torch.no_grad()
def run_inference(model, loader, mean: torch.Tensor, std: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward the whole loader; (predictions, ground truth) as [n_chunks, k, 3]
    arrays in raw units (the model regresses standardised actions).
    """
    model.eval()
    predictions, ground_truth = [], []
    for batch in loader:
        latents = chunk_grid(batch).to(DEVICE, non_blocking=True)
        with autocast(DEVICE):
            predicted_standardised = model(latents)  # [B, k, 3]
        predictions.append((predicted_standardised.float() * std + mean).cpu().numpy())
        ground_truth.append(batch["future_ego_motions"].numpy())
    return np.concatenate(predictions), np.concatenate(ground_truth)


# ----------------------------------------------------------------------------- metrics

def smooth_l1(error: np.ndarray, beta: float = 1.0) -> float:
    """Mean Smooth-L1 (Huber-like, matching F.smooth_l1_loss) of an error array."""
    absolute = np.abs(error)
    return float(np.where(absolute < beta, 0.5 * absolute**2 / beta, absolute - 0.5 * beta).mean())


def metrics_report(predictions: np.ndarray, ground_truth: np.ndarray,
                   train_mean: np.ndarray, train_median: np.ndarray) -> dict[str, dict[str, float]]:
    """
    Per-component MAE / R^2 / Pearson r for the IDM and for the constant
    baselines: training median for MAE (its optimal constant), training mean
    for R^2 (likewise). All chunk steps are pooled; values in raw units
    (metres / radians).
    """
    predictions = predictions.reshape(-1, 3).astype(np.float64)
    ground_truth = ground_truth.reshape(-1, 3).astype(np.float64)

    report = {}
    for component, name in enumerate(COMPONENT_NAMES):
        predicted = predictions[:, component]
        actual = ground_truth[:, component]
        mean_baseline = float(train_mean[component])
        median_baseline = float(train_median[component])
        total_ss = ((actual - actual.mean()) ** 2).sum()
        report[name] = {
            "mae": float(np.abs(predicted - actual).mean()),
            "baseline_mae": float(np.abs(median_baseline - actual).mean()),
            "r2": float(1.0 - ((actual - predicted) ** 2).sum() / total_ss),
            "baseline_r2": float(1.0 - ((actual - mean_baseline) ** 2).sum() / total_ss),
            "pearson": float(np.corrcoef(predicted, actual)[0, 1]),
        }
    return report


def print_report(report: dict[str, dict[str, float]]) -> None:
    print(f"\n{'component':<10} {'MAE':>14} {'base MAE':>14} {'R^2':>8} {'base R^2':>9} {'pearson':>8}")
    for name, metrics in report.items():
        scale, unit = DISPLAY_SCALE[name], DISPLAY_UNIT[name]
        print(
            f"{name:<10} {metrics['mae'] * scale:>10.4f} {unit:<3} "
            f"{metrics['baseline_mae'] * scale:>10.4f} {unit:<3} "
            f"{metrics['r2']:>8.4f} {metrics['baseline_r2']:>9.4f} {metrics['pearson']:>8.4f}"
        )
    if all(
        metrics["mae"] < metrics["baseline_mae"] and metrics["r2"] > metrics["baseline_r2"]
        for metrics in report.values()
    ):
        print("\nsuccess: beats the constant baselines on every component")
    else:
        losing = [
            name for name, metrics in report.items()
            if metrics["mae"] >= metrics["baseline_mae"] or metrics["r2"] <= metrics["baseline_r2"]
        ]
        print(f"\nFAILURE: does not beat the constant baselines on: {', '.join(losing)}")


# ----------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Probe ego-motion recoverability from V-JEPA features")
    parser.add_argument("--horizon", type=int, default=1, help="action chunk length k (plan: 1 and 4)")
    parser.add_argument("--max-iters", type=int, default=20_000)
    parser.add_argument("--grad-accum", type=int, default=1)  # eff. batch = grad_accum * batch_size
    parser.add_argument("--val-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="defaults to checkpoints_idm/idm_k<horizon>_best.pt")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; report test metrics from the saved checkpoint")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or OUTPUTS_DIR / "checkpoints_idm" / f"idm_k{args.horizon}_best.pt"

    loaders = KITTIRolloutLoaders(
        training_sequences=TRAIN_SEQUENCES,
        validation_sequences=VALIDATION_SEQUENCES,
        test_sequences=TEST_SEQUENCES,
        context_length=1,  # k+1 frames total: 1 "context" + k "future"
        future_length=args.horizon,
        frame_stride=args.frame_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(loaders)

    if args.eval_only:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        config = checkpoint["config"]
        model = create_idm(
            feature_dim=config["feature_dim"],
            spatial_h=config["grid_hw"][0],
            spatial_w=config["grid_hw"][1],
            horizon=config["horizon"],
            patch_size=config["patch_size"],
        ).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded {checkpoint_path} (iter {checkpoint['iteration']}, "
              f"val loss {checkpoint['validation_metrics']['loss']:.4f})")
        train_mean = np.array(config["action_mean"], dtype=np.float32)
        train_std = np.array(config["action_std"], dtype=np.float32)
        if "action_median" in config:
            train_median = np.array(config["action_median"], dtype=np.float32)
        else:  # checkpoint predates the median baseline: recompute (reporting only)
            _, _, train_median = action_statistics(loaders.train_dataset)
        mean = torch.tensor(train_mean, device=DEVICE)
        std = torch.tensor(train_std, device=DEVICE)
        predictions, ground_truth = run_inference(model, loaders.test, mean, std)
        print_report(metrics_report(predictions, ground_truth, train_mean, train_median))
        return

    # Step 1: training-set statistics. Printing them makes the imbalance the
    # standardisation corrects for visible (forward >> left, yaw).
    train_mean, train_std, train_median = action_statistics(loaders.train_dataset)
    for component, name in enumerate(COMPONENT_NAMES):
        scale, unit = DISPLAY_SCALE[name], DISPLAY_UNIT[name]
        print(f"train action stats | {name:<8} mean {train_mean[component] * scale:>8.4f} {unit:<3} "
              f"std {train_std[component] * scale:>8.4f} {unit:<3} "
              f"median {train_median[component] * scale:>8.4f} {unit}")
    mean = torch.tensor(train_mean, device=DEVICE)
    std = torch.tensor(train_std, device=DEVICE)

    wandb.init(
        project="jepa-drive-wm",
        job_type="idm",
        config={
            **vars(args),
            "checkpoint": str(checkpoint_path),
            "step_seconds": loaders.step_seconds,
            "action_mean": train_mean.tolist(),
            "action_std": train_std.tolist(),
            "action_median": train_median.tolist(),
        },
    )
    wandb.define_metric("iter")
    wandb.define_metric("train/*", step_metric="iter")
    wandb.define_metric("val/*", step_metric="iter")

    # Step 2: train the IDM with standardised Smooth-L1.
    model = create_idm(
        feature_dim=VJEPA_BASE_EMBED_DIM,
        spatial_h=GRID_H,
        spatial_w=GRID_W,
        horizon=args.horizon,
        patch_size=args.patch_size,
    ).to(DEVICE)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_opt_steps = args.max_iters // args.grad_accum
    scheduler = build_warmup_cosine(optimizer, total_opt_steps, int(args.warmup_frac * total_opt_steps))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    train_batches = infinite(loaders.train)
    running_loss = 0.0
    grad_norm = torch.zeros(())
    optimizer.zero_grad(set_to_none=True)
    model.train()
    start = time.time()

    for iteration in range(1, args.max_iters + 1):
        batch = next(train_batches)
        latents = chunk_grid(batch).to(DEVICE, non_blocking=True)
        target = (batch["future_ego_motions"].to(DEVICE, non_blocking=True) - mean) / std

        with autocast(DEVICE):
            predicted = model(latents)  # [B, k, 3] standardised
        loss = F.smooth_l1_loss(predicted.float(), target)
        (loss / args.grad_accum).backward()

        if iteration % args.grad_accum == 0:
            # clip_grad_norm_ returns the pre-clip norm: a free health metric.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += loss.item()

        if iteration % args.log_every == 0:
            rate = iteration / (time.time() - start)
            mean_loss = running_loss / args.log_every
            running_loss = 0.0
            print(f"iter {iteration:6d}/{args.max_iters} | loss {mean_loss:.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {rate:.1f} it/s", flush=True)
            wandb.log({
                "iter": iteration,
                "train/loss": mean_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": grad_norm.item(),
            })

        if iteration % args.val_every == 0 or iteration == args.max_iters:
            predictions, ground_truth = run_inference(model, loaders.validation, mean, std)
            model.train()
            # Same quantity as the training loss: Smooth-L1 in standardised action space.
            validation_loss = smooth_l1((predictions - ground_truth) / train_std)
            print(f"  [val] iter {iteration} | loss {validation_loss:.4f}", flush=True)
            wandb.log({"iter": iteration, "val/loss": validation_loss})
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": model.state_dict(),
                        "validation_metrics": {"loss": validation_loss},
                        "config": {
                            "horizon": args.horizon,
                            "patch_size": args.patch_size,
                            "frame_stride": args.frame_stride,
                            "step_seconds": loaders.step_seconds,
                            "feature_dim": VJEPA_BASE_EMBED_DIM,
                            "grid_hw": list(VJEPA_BASE_GRID_HW),
                            "action_mean": train_mean.tolist(),
                            "action_std": train_std.tolist(),
                            "action_median": train_median.tolist(),
                        },
                    },
                    checkpoint_path,
                )
                wandb.summary["best_val_loss"] = validation_loss
                wandb.summary["best_iter"] = iteration
                print(f"          -> saved new best model to {checkpoint_path}")

    # Test with the best-validation model, not whatever the last iteration left.
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE)["model_state_dict"])
    predictions, ground_truth = run_inference(model, loaders.test, mean, std)
    report = metrics_report(predictions, ground_truth, train_mean, train_median)
    print_report(report)
    # Single points, not curves: they belong in the summary, not the timeline.
    wandb.summary.update({
        f"test/{name}/{metric}": value
        for name, metrics in report.items()
        for metric, value in metrics.items()
    })
    wandb.finish()


if __name__ == "__main__":
    main()
