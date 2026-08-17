"""
Evaluate the trained depth decoder on the held-out KITTI test sequences.

- load the best checkpoint
- report depth metrics via DepthEvaluator:
      AbsRel, RMSE, delta1 -- over all valid pixels, over non-sky pixels
      (sky/ignore group excluded; sky pseudolabel depth is noise), and over
      vehicle pixels only, the latter as a proxy for vehicle localisation
      (both masks from the OneFormer semantics GT)
- save figures: RGB | GT + prediction (linear and log depth scales, shared
  colour bars) | absolute and relative error maps on fixed scales
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from jepa_drive_wm.data.data_interface_dense import KITTIDepthDataset
from jepa_drive_wm.models.dense_decoders.depth_decoder import DepthDecoder
from jepa_drive_wm.train.train_depth import (
    CHECKPOINT_PATH,
    DEVICE,
    MAX_DEPTH,
    MIN_DEPTH,
    TEST_SEQUENCES,
    predict,
)
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.viz.visualiser import CLASS_NAMES, CLASS_TO_GROUP, GROUP_NAMES, NUM_CLASSES

FIGURES_DIR = OUTPUTS_DIR / "evals_depth"

DELTA1_THRESHOLD = 1.25  # delta1 = fraction of pixels with max(pred/gt, gt/pred) < 1.25

# Cityscapes vehicle classes, masked from the OneFormer semantics GT. Person and
# rider are deliberately excluded: this is a vehicle-localisation proxy.
VEHICLE_CLASS_NAMES = ("car", "truck", "bus", "train", "motorcycle", "bicycle")

# Class id (0..18) -> is vehicle. Sized 256 so IGNORE_INDEX maps to background.
VEHICLE_LUT = np.zeros(256, dtype=bool)
for _name in VEHICLE_CLASS_NAMES:
    VEHICLE_LUT[CLASS_NAMES.index(_name)] = True

# Class id -> is sky/ignore (the coarse planning group), used to exclude sky
# from the non-sky metrics and the error maps: FoundationStereo sky depth is
# essentially noise and would otherwise dominate the error. Anything outside
# 0..18 (255 = unlabeled) also counts as ignore.
SKY_IGNORE_LUT = np.ones(256, dtype=bool)
SKY_IGNORE_LUT[:NUM_CLASSES] = CLASS_TO_GROUP == GROUP_NAMES.index("sky / ignore")


class ModelPredictions:
    """
    The trained decoder's metric depth maps over a dataset, indexable like a
    list: predictions[i] is the (H, W) float32 depth in metres for dataset[i].

    Lazy and uncached -- each access is one forward pass -- so a metrics pass
    streams over ~2k frames instead of holding every full-resolution map in
    memory, and DepthEvaluator stays decoupled from the model: it accepts
    anything indexable that is frame-aligned with its test set.
    """

    def __init__(self, model: DepthDecoder, dataset: KITTIDepthDataset) -> None:
        self.model = model
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, item: int) -> np.ndarray:
        sample = self.dataset[item]
        depth = predict(self.model, sample["features"][None].to(DEVICE), sample["target"].shape)
        return depth.cpu().numpy()


def _valid_mask(target: np.ndarray) -> np.ndarray:
    """Pixels where the pseudolabel is trustworthy and inside the depth range
    (numpy twin of train_depth._valid_mask)."""
    return np.isfinite(target) & (target > MIN_DEPTH) & (target < MAX_DEPTH)


class DepthEvaluator:
    """
    Given the test set and frame-aligned predictions of it (from the trained
    model), compute depth metrics against the FoundationStereo pseudolabels and
    qualitative figures.

    All metrics come from a single pass over the test set that accumulates
    sufficient statistics (error sums + pixel counts) for three pixel scopes,
    masked from the OneFormer semantics GT: "all" valid pixels, "non-sky"
    (sky/ignore group excluded) and "vehicle" (a vehicle-localisation proxy).
    The pass runs
    lazily on the first metric call and is cached. Convention: pixels are
    pooled over the whole test set (dataset-level, not mean-of-frames -- note
    some benchmarks average per-image instead).
    """

    def __init__(self, test_set: KITTIDepthDataset, predicted_test_set) -> None:
        self.test_set = test_set
        self.predictions = predicted_test_set
        self.figures_dir = FIGURES_DIR
        self._stats: dict | None = None

    # ------------------------------------------------------------------ accumulation

    def _accumulate(self) -> dict:
        """One pass over the test set: error sums and pixel counts per scope."""
        if self._stats is not None:
            return self._stats

        stats = {
            scope: {"absrel_sum": 0.0, "squared_error_sum": 0.0, "delta1_hits": 0, "pixels": 0}
            for scope in ("all", "non-sky", "vehicle")
        }

        for item, (sequence_nr, frame_index) in enumerate(self.test_set.index):
            sequence = self.test_set.sequences[sequence_nr]
            gt = sequence.get_depth(frame_index)          # (H, W) metres
            pred = self.predictions[item]                 # (H, W) metres
            semantics = sequence.get_semantics(frame_index)
            valid = _valid_mask(gt)
            non_sky = valid & ~SKY_IGNORE_LUT[semantics]
            vehicle = valid & VEHICLE_LUT[semantics]

            for scope, mask in (("all", valid), ("non-sky", non_sky), ("vehicle", vehicle)):
                g, p = gt[mask], pred[mask]
                ratio = np.maximum(p / g, g / p)  # g > MIN_DEPTH and p > 0 by construction
                stats[scope]["absrel_sum"] += float((np.abs(p - g) / g).sum())
                stats[scope]["squared_error_sum"] += float(((p - g) ** 2).sum())
                stats[scope]["delta1_hits"] += int((ratio < DELTA1_THRESHOLD).sum())
                stats[scope]["pixels"] += int(mask.sum())

        self._stats = stats
        return self._stats

    def _absrel(self, scope: str) -> float:
        stats = self._accumulate()[scope]
        return stats["absrel_sum"] / stats["pixels"] if stats["pixels"] else float("nan")

    def _rmse(self, scope: str) -> float:
        stats = self._accumulate()[scope]
        return float(np.sqrt(stats["squared_error_sum"] / stats["pixels"])) if stats["pixels"] else float("nan")

    def _d1(self, scope: str) -> float:
        stats = self._accumulate()[scope]
        return stats["delta1_hits"] / stats["pixels"] if stats["pixels"] else float("nan")

    # ------------------------------------------------------------------ metrics

    def calculate_absrel(self) -> float:
        """Mean |pred - gt| / gt over all valid pixels."""
        return self._absrel("all")

    def calculate_rmse(self) -> float:
        """Root mean squared error in metres over all valid pixels."""
        return self._rmse("all")

    def calculate_d1(self) -> float:
        """Threshold accuracy: fraction of valid pixels with max(pred/gt, gt/pred) < 1.25."""
        return self._d1("all")

    def calculate_non_sky_depth_error(self) -> dict[str, float]:
        """
        AbsRel / RMSE / delta1 excluding the sky/ignore planning group (sky
        class + unlabeled). FoundationStereo sky depth is essentially noise, so
        this scope is the honest summary of geometry the decoder should get right.
        """
        return {"absrel": self._absrel("non-sky"), "rmse": self._rmse("non-sky"), "d1": self._d1("non-sky")}

    def calculate_vehicle_pixel_depth_error(self) -> dict[str, float]:
        """
        AbsRel / RMSE / delta1 restricted to vehicle pixels (VEHICLE_CLASS_NAMES
        in the OneFormer semantics GT). Depth on vehicles is what turns "there
        is a car" into "there is a car 12 m ahead", so this is the
        vehicle-localisation proxy.
        """
        return {"absrel": self._absrel("vehicle"), "rmse": self._rmse("vehicle"), "d1": self._d1("vehicle")}

    # ------------------------------------------------------------------ qualitative

    def visualise_depth(self, viz_every: int = 100) -> None:
        """
        Every `viz_every` frames, one PNG with 7 rows:

            RGB | GT + prediction on a linear depth scale | GT + prediction on a
            log depth scale | absolute error | absolute relative error
            (error maps exclude the sky/ignore group, whose pseudolabel depth
            is noise that overwhelms the real signal)

        The log view expands the near range that the linear 0.5-80 m scale
        compresses (where driving decisions live); GT and prediction share one
        normalisation and colour bar per pair so colours are comparable. Error
        maps use fixed scales (0-10 m, 0-0.5) so figures are comparable across
        frames and checkpoints; invalid-GT pixels render white.
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        for item in range(0, len(self.test_set), viz_every):
            sequence_nr, frame_index = self.test_set.index[item]
            sequence = self.test_set.sequences[sequence_nr]
            gt = sequence.get_depth(frame_index)
            pred = self.predictions[item]

            valid = _valid_mask(gt)
            gt_display = np.where(valid, gt, np.nan)
            # Sky depth is pseudolabel noise and would overwhelm the error maps.
            error_scope = valid & ~SKY_IGNORE_LUT[sequence.get_semantics(frame_index)]
            absolute_error = np.where(error_scope, np.abs(pred - gt), np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                relative_error = np.where(error_scope, np.abs(pred - gt) / gt, np.nan)

            fig = plt.figure(figsize=(11, 20), constrained_layout=True)
            grid = fig.add_gridspec(7, 2, width_ratios=[1, 0.025])
            axes = [fig.add_subplot(grid[row, 0]) for row in range(7)]
            rgb_ax, gt_linear_ax, pred_linear_ax, gt_log_ax, pred_log_ax, abs_ax, rel_ax = axes

            rgb_ax.imshow(sequence.get_image(frame_index))
            rgb_ax.set_title(f"seq {sequence_nr:02d} frame {frame_index}")

            linear = {"cmap": "plasma", "vmin": MIN_DEPTH, "vmax": MAX_DEPTH}
            image = gt_linear_ax.imshow(gt_display, **linear)
            gt_linear_ax.set_title("FoundationStereo depth (linear)")
            pred_linear_ax.imshow(pred, **linear)
            pred_linear_ax.set_title("predicted depth (linear)")
            fig.colorbar(image, cax=fig.add_subplot(grid[1:3, 1]), label="Depth (m)")

            log_norm = LogNorm(vmin=MIN_DEPTH, vmax=MAX_DEPTH)
            image = gt_log_ax.imshow(gt_display, cmap="plasma", norm=log_norm)
            gt_log_ax.set_title("FoundationStereo depth (log)")
            pred_log_ax.imshow(pred, cmap="plasma", norm=log_norm)
            pred_log_ax.set_title("predicted depth (log)")
            colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[3:5, 1]), label="Depth (m)")
            colorbar.set_ticks([1, 2, 5, 10, 20, 40, 80],
                               labels=["1", "2", "5", "10", "20", "40", "80"])

            image = abs_ax.imshow(absolute_error, cmap="inferno", vmin=0, vmax=10)
            abs_ax.set_title("absolute error (non-sky)")
            fig.colorbar(image, cax=fig.add_subplot(grid[5, 1]), extend="max",
                         label="|pred - gt| (m)")

            image = rel_ax.imshow(relative_error, cmap="inferno", vmin=0, vmax=0.5)
            rel_ax.set_title("absolute relative error (non-sky)")
            fig.colorbar(image, cax=fig.add_subplot(grid[6, 1]), extend="max",
                         label="|pred - gt| / gt")

            for ax in axes:
                ax.axis("off")
            fig.savefig(self.figures_dir / f"seq{sequence_nr:02d}_frame{frame_index:06d}.png", dpi=150)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained depth decoder")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--viz-every", type=int, default=100,
                        help="save a figure every N test frames (0 disables)")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    config = checkpoint.get("config", {})
    model = DepthDecoder(
        bins_strategy=config.get("bins_strategy", "linear"),
        norm_strategy=config.get("norm_strategy", "linear"),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"loaded {args.checkpoint} (iter {checkpoint['iteration']}, "
          f"val total {checkpoint['validation_metrics']['total']:.4f})")

    dataset = KITTIDepthDataset(TEST_SEQUENCES)

    evaluator = DepthEvaluator(dataset, ModelPredictions(model, dataset))
    print(f"\nAbsRel        {evaluator.calculate_absrel():.4f}")
    print(f"RMSE          {evaluator.calculate_rmse():.4f} m")
    print(f"delta1        {evaluator.calculate_d1():.4f}")
    non_sky = evaluator.calculate_non_sky_depth_error()
    print("\nnon-sky pixels only (sky/ignore group excluded):")
    print(f"AbsRel        {non_sky['absrel']:.4f}")
    print(f"RMSE          {non_sky['rmse']:.4f} m")
    print(f"delta1        {non_sky['d1']:.4f}")
    vehicle = evaluator.calculate_vehicle_pixel_depth_error()
    print("\nvehicle pixels only (" + ", ".join(VEHICLE_CLASS_NAMES) + "):")
    print(f"AbsRel        {vehicle['absrel']:.4f}")
    print(f"RMSE          {vehicle['rmse']:.4f} m")
    print(f"delta1        {vehicle['d1']:.4f}")

    if args.viz_every:
        evaluator.visualise_depth(args.viz_every)
        print(f"figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
