"""
Evaluate the trained depth decoder on the held-out KITTI test sequences
(SPLIT_V1.test_sequences, see data/splits.py).

    PYTHONPATH=src python -m jepa_drive_wm.evals.depth_evaluator [--checkpoint ...]

- load the best checkpoint (built with the bins/norm strategies it records)
- metrics via DepthEvaluator, all from one streaming pass over the test frames:
      AbsRel, RMSE, delta1 -- over all valid pixels, over non-sky pixels
      (sky/ignore group excluded; sky pseudolabel depth is noise), and over
      vehicle pixels only, the latter as a proxy for vehicle localisation
      (both masks from the OneFormer semantics GT);
      the same three scopes per test sequence (01 highway / 07 urban / 09 rural
      behave very differently), and AbsRel / RMSE / delta1 binned by GT depth
      range (non-sky scope) to show where the error lives
- outputs in OUTPUTS_DIR/evals_depth:
      metrics.json, metrics.md                       all numbers above
      error_by_depth_range.png                       AbsRel + delta1 per GT depth bin
      seqXX_frameNNNNNN.png (every --viz-every)      RGB | GT + prediction (linear
          and log depth scales, shared colour bars) | absolute and relative
          error maps on fixed scales

The metric definitions live in DepthMetricAccumulator so the world-model
evaluator can score decoded rollouts with exactly the same code.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from jepa_drive_wm.data.data_interface_dense import KITTIDepthDataset
from jepa_drive_wm.data.splits import SPLIT_V1
from jepa_drive_wm.evals.common import (
    DEPTH_CHECKPOINT,
    DEVICE,
    SERIES,
    describe_checkpoint,
    finish_figure,
    load_depth_decoder,
    markdown_table,
    new_figure,
    style_axes,
    write_metrics_json,
)
from jepa_drive_wm.models.dense_decoders.depth_decoder import DepthDecoder
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.train.train_depth import MAX_DEPTH, MIN_DEPTH, predict
from jepa_drive_wm.viz.visualiser import CLASS_NAMES, CLASS_TO_GROUP, GROUP_NAMES, NUM_CLASSES

FIGURES_DIR = OUTPUTS_DIR / "evals_depth"
TEST_SEQUENCES = list(SPLIT_V1.test_sequences)

DELTA1_THRESHOLD = 1.25  # delta1 = fraction of pixels with max(pred/gt, gt/pred) < 1.25

# GT depth bins (metres) for the error-by-range breakdown. Non-sky scope; the
# last edge is MAX_DEPTH so the bins tile the valid range exactly.
RANGE_EDGES = (MIN_DEPTH, 2.0, 5.0, 10.0, 20.0, 40.0, MAX_DEPTH)

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

SCOPES = ("all", "non-sky", "vehicle")


def _valid_mask(target: np.ndarray) -> np.ndarray:
    """Pixels where the pseudolabel is trustworthy and inside the depth range
    (numpy twin of train_depth._valid_mask)."""
    return np.isfinite(target) & (target > MIN_DEPTH) & (target < MAX_DEPTH)


class DepthMetricAccumulator:
    """
    Streaming depth metrics against FoundationStereo pseudolabels.

    update() is fed one frame at a time (GT depth, predicted depth, OneFormer
    semantics for the masks) and keeps sufficient statistics (error sums +
    pixel counts) for the three pixel scopes -- "all" valid pixels, "non-sky"
    (sky/ignore group excluded) and "vehicle" (vehicle-localisation proxy) --
    plus the same statistics per GT depth bin on the non-sky scope.

    Convention: pixels are pooled over everything fed in (dataset-level, not
    mean-of-frames -- note some benchmarks average per-image instead).
    """

    def __init__(self, range_edges: tuple[float, ...] = RANGE_EDGES) -> None:
        self.range_edges = tuple(range_edges)
        self._scope = {scope: self._empty() for scope in SCOPES}
        self._range = [self._empty() for _ in range(len(self.range_edges) - 1)]
        self.frames = 0

    @staticmethod
    def _empty() -> dict:
        return {"absrel_sum": 0.0, "squared_error_sum": 0.0, "delta1_hits": 0, "pixels": 0}

    @staticmethod
    def _add(stats: dict, gt: np.ndarray, pred: np.ndarray) -> None:
        ratio = np.maximum(pred / gt, gt / pred)  # gt > MIN_DEPTH and pred > 0 by construction
        stats["absrel_sum"] += float((np.abs(pred - gt) / gt).sum())
        stats["squared_error_sum"] += float(((pred - gt) ** 2).sum())
        stats["delta1_hits"] += int((ratio < DELTA1_THRESHOLD).sum())
        stats["pixels"] += int(gt.size)

    def update(self, gt: np.ndarray, pred: np.ndarray, semantics: np.ndarray) -> None:
        """gt, pred: (H, W) metres; semantics: (H, W) Cityscapes ids (255 = unlabeled)."""
        valid = _valid_mask(gt)
        non_sky = valid & ~SKY_IGNORE_LUT[semantics]
        vehicle = valid & VEHICLE_LUT[semantics]
        for scope, mask in (("all", valid), ("non-sky", non_sky), ("vehicle", vehicle)):
            self._add(self._scope[scope], gt[mask], pred[mask])

        g, p = gt[non_sky], pred[non_sky]
        # digitize with the interior edges: bin i holds edges[i] <= g < edges[i+1]
        bins = np.digitize(g, self.range_edges[1:-1])
        for i, stats in enumerate(self._range):
            in_bin = bins == i
            if in_bin.any():
                self._add(stats, g[in_bin], p[in_bin])
        self.frames += 1

    @staticmethod
    def _metrics(stats: dict) -> dict[str, float]:
        n = stats["pixels"]
        if not n:
            return {"absrel": float("nan"), "rmse": float("nan"), "d1": float("nan"), "pixels": 0}
        return {
            "absrel": stats["absrel_sum"] / n,
            "rmse": float(np.sqrt(stats["squared_error_sum"] / n)),
            "d1": stats["delta1_hits"] / n,
            "pixels": n,
        }

    def metrics(self, scope: str = "all") -> dict[str, float]:
        """{'absrel', 'rmse', 'd1', 'pixels'} for one scope."""
        return self._metrics(self._scope[scope])

    def summary(self) -> dict[str, dict[str, float]]:
        """All three scopes at once."""
        return {scope: self.metrics(scope) for scope in SCOPES}

    def range_table(self) -> list[dict]:
        """Per GT depth bin (non-sky scope): the three metrics + the bin's pixel share."""
        total = sum(s["pixels"] for s in self._range)
        rows = []
        for (lo, hi), stats in zip(zip(self.range_edges[:-1], self.range_edges[1:]), self._range):
            row = {"lo": lo, "hi": hi, **self._metrics(stats)}
            row["pixel_share"] = stats["pixels"] / total if total else float("nan")
            rows.append(row)
        return rows


class ModelPredictions:
    """
    The trained decoder's metric depth maps over a dataset, indexable like a
    list: predictions[i] is the (H, W) float32 depth in metres for dataset[i].

    Lazy and uncached -- each access is one forward pass -- so a metrics pass
    streams over ~4k frames instead of holding every full-resolution map in
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


class DepthEvaluator:
    """
    Given the test set and frame-aligned predictions of it (from the trained
    model), compute depth metrics against the FoundationStereo pseudolabels and
    qualitative figures.

    One streaming pass over the test set (run lazily on the first metric call
    and cached) feeds a dataset-level DepthMetricAccumulator and one per
    sequence. `max_frames` caps the pass for smoke tests.
    """

    def __init__(self, test_set: KITTIDepthDataset, predicted_test_set,
                 figures_dir: Path = FIGURES_DIR, max_frames: int | None = None) -> None:
        self.test_set = test_set
        self.predictions = predicted_test_set
        self.figures_dir = figures_dir
        self.max_frames = max_frames
        self._overall: DepthMetricAccumulator | None = None
        self._per_sequence: dict[int, DepthMetricAccumulator] = {}

    # ------------------------------------------------------------------ accumulation

    def _items(self) -> list[int]:
        n = len(self.test_set) if self.max_frames is None else min(self.max_frames, len(self.test_set))
        return list(range(n))

    def _accumulate(self) -> DepthMetricAccumulator:
        """One pass over the test set: error sums and pixel counts per scope."""
        if self._overall is not None:
            return self._overall

        overall = DepthMetricAccumulator()
        per_sequence: dict[int, DepthMetricAccumulator] = {}
        for item in self._items():
            sequence_nr, frame_index = self.test_set.index[item]
            sequence = self.test_set.sequences[sequence_nr]
            gt = sequence.get_depth(frame_index)          # (H, W) metres
            pred = self.predictions[item]                 # (H, W) metres
            semantics = sequence.get_semantics(frame_index)
            overall.update(gt, pred, semantics)
            per_sequence.setdefault(sequence_nr, DepthMetricAccumulator()).update(gt, pred, semantics)

        self._overall, self._per_sequence = overall, per_sequence
        return overall

    # ------------------------------------------------------------------ metrics

    def calculate_absrel(self) -> float:
        """Mean |pred - gt| / gt over all valid pixels."""
        return self._accumulate().metrics("all")["absrel"]

    def calculate_rmse(self) -> float:
        """Root mean squared error in metres over all valid pixels."""
        return self._accumulate().metrics("all")["rmse"]

    def calculate_d1(self) -> float:
        """Threshold accuracy: fraction of valid pixels with max(pred/gt, gt/pred) < 1.25."""
        return self._accumulate().metrics("all")["d1"]

    def calculate_non_sky_depth_error(self) -> dict[str, float]:
        """
        AbsRel / RMSE / delta1 excluding the sky/ignore planning group (sky
        class + unlabeled). FoundationStereo sky depth is essentially noise, so
        this scope is the honest summary of geometry the decoder should get right.
        """
        return self._accumulate().metrics("non-sky")

    def calculate_vehicle_pixel_depth_error(self) -> dict[str, float]:
        """
        AbsRel / RMSE / delta1 restricted to vehicle pixels (VEHICLE_CLASS_NAMES
        in the OneFormer semantics GT). Depth on vehicles is what turns "there
        is a car" into "there is a car 12 m ahead", so this is the
        vehicle-localisation proxy.
        """
        return self._accumulate().metrics("vehicle")

    def per_sequence_metrics(self) -> dict[int, dict[str, dict[str, float]]]:
        """{sequence_nr: {scope: {absrel, rmse, d1, pixels}}}."""
        self._accumulate()
        return {seq: acc.summary() for seq, acc in sorted(self._per_sequence.items())}

    def error_by_depth_range(self) -> list[dict]:
        """Non-sky AbsRel / RMSE / delta1 per GT depth bin (RANGE_EDGES)."""
        return self._accumulate().range_table()

    def all_metrics(self) -> dict:
        """Everything, ready for metrics.json."""
        overall = self._accumulate()
        return {
            "frames": overall.frames,
            "overall": overall.summary(),
            "per_sequence": {f"{seq:02d}": m for seq, m in self.per_sequence_metrics().items()},
            "by_depth_range": self.error_by_depth_range(),
        }

    # ------------------------------------------------------------------ qualitative

    def plot_error_by_depth_range(self, path: Path | None = None) -> Path:
        """Two panels: AbsRel and delta1 per GT depth bin (non-sky), pixel share as bar labels."""
        rows = self.error_by_depth_range()
        labels = [f"{r['lo']:g}-{r['hi']:g} m" for r in rows]
        x = np.arange(len(rows))
        fig, axes = new_figure(ncols=2, width=6.0, height=4.0)
        for ax, key, title in zip(axes[0], ("absrel", "d1"),
                                  ("AbsRel by GT depth (non-sky)", "delta1 by GT depth (non-sky)")):
            values = [r[key] for r in rows]
            ax.bar(x, values, width=0.7, color=SERIES[0], linewidth=0)
            for xi, value, row in zip(x, values, rows):
                if np.isfinite(value):
                    ax.text(xi, value, f"{value:.3f}\n({row['pixel_share']:.0%} px)",
                            ha="center", va="bottom", fontsize=7.5, color="#52514e")
            ax.set_xticks(x, labels, rotation=0, fontsize=8)
            style_axes(ax, title=title, ylabel=key)
            if key == "d1":
                ax.set_ylim(0, 1.05)
        path = path or self.figures_dir / "error_by_depth_range.png"
        finish_figure(fig, path)
        return path

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
        for item in self._items()[::viz_every]:
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


# ----------------------------------------------------------------------------- reporting

def metrics_markdown(metrics: dict, checkpoint_line: str) -> str:
    """metrics.md: the console overview as tables."""
    overall = metrics["overall"]
    scope_rows = [[scope, m["absrel"], m["rmse"], m["d1"], m["pixels"]]
                  for scope, m in overall.items()]
    seq_rows = [[seq, m["non-sky"]["absrel"], m["non-sky"]["rmse"], m["non-sky"]["d1"],
                 m["vehicle"]["absrel"], m["vehicle"]["d1"], m["all"]["absrel"]]
                for seq, m in metrics["per_sequence"].items()]
    range_rows = [[f"{r['lo']:g}-{r['hi']:g} m", r["absrel"], r["rmse"], r["d1"],
                   r["pixel_share"]] for r in metrics["by_depth_range"]]
    return "\n".join([
        "# Depth decoder -- test set (SPLIT_V1: sequences "
        + ", ".join(f"{s:02d}" for s in TEST_SEQUENCES) + ")",
        "",
        f"checkpoint: `{checkpoint_line}`  ",
        f"frames: {metrics['frames']}; pixels pooled over the test set (dataset-level, not mean-of-frames)",
        "",
        "## Overall",
        "",
        markdown_table(["scope", "AbsRel", "RMSE (m)", "delta1", "pixels"], scope_rows),
        "",
        "scopes: all valid pixels (" + f"{MIN_DEPTH:g} < gt < {MAX_DEPTH:g} m); non-sky = sky/ignore "
        "group excluded; vehicle = " + ", ".join(VEHICLE_CLASS_NAMES),
        "",
        "## Per sequence",
        "",
        markdown_table(["seq", "non-sky AbsRel", "non-sky RMSE (m)", "non-sky delta1",
                        "vehicle AbsRel", "vehicle delta1", "all AbsRel"], seq_rows),
        "",
        "## By GT depth range (non-sky)",
        "",
        markdown_table(["GT depth", "AbsRel", "RMSE (m)", "delta1", "pixel share"], range_rows),
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained depth decoder on the test split")
    parser.add_argument("--checkpoint", type=Path, default=DEPTH_CHECKPOINT)
    parser.add_argument("--viz-every", type=int, default=100,
                        help="save a figure every N test frames (0 disables)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="only score the first N test frames (smoke tests)")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    model, checkpoint = load_depth_decoder(args.checkpoint)
    checkpoint_line = describe_checkpoint(args.checkpoint, checkpoint)
    print(f"loaded {checkpoint_line}")

    dataset = KITTIDepthDataset(TEST_SEQUENCES)
    evaluator = DepthEvaluator(dataset, ModelPredictions(model, dataset),
                               figures_dir=args.figures_dir, max_frames=args.max_frames)

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

    metrics = evaluator.all_metrics()
    metrics["checkpoint"] = checkpoint_line
    metrics["test_sequences"] = TEST_SEQUENCES
    print("\nper sequence (non-sky AbsRel / delta1):")
    for seq, m in metrics["per_sequence"].items():
        print(f"  seq {seq}: AbsRel {m['non-sky']['absrel']:.4f}  delta1 {m['non-sky']['d1']:.4f}  "
              f"({m['non-sky']['pixels'] / 1e6:.1f}M px)")
    print("\nby GT depth range (non-sky):")
    for r in metrics["by_depth_range"]:
        print(f"  {r['lo']:>4g}-{r['hi']:<4g} m: AbsRel {r['absrel']:.4f}  RMSE {r['rmse']:.3f} m  "
              f"delta1 {r['d1']:.4f}  ({r['pixel_share']:.1%} of px)")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(metrics, args.figures_dir / "metrics.json")
    (args.figures_dir / "metrics.md").write_text(metrics_markdown(metrics, checkpoint_line))
    evaluator.plot_error_by_depth_range()
    if args.viz_every:
        evaluator.visualise_depth(args.viz_every)
    print(f"\nmetrics + figures saved to {args.figures_dir}")


if __name__ == "__main__":
    main()
