"""
Evaluate the trained semantic decoder on the held-out KITTI test sequences
(SPLIT_V1.test_sequences, see data/splits.py).

    PYTHONPATH=src python -m jepa_drive_wm.evals.semantics_evaluator [--checkpoint ...]

- load the best checkpoint
- metrics via SemanticsEvaluator, all from one streaming pass over the test frames:
      per-class IoU, 19-class mIoU, pixel accuracy, non-sky planning-group mIoU,
      drivable IoU, drivable Boundary IoU, traffic-participant IoU, car IoU;
      the headline numbers again per test sequence
- outputs in OUTPUTS_DIR/evals_semantics:
      metrics.json, metrics.md                       all numbers above
      per_class_iou.png                              horizontal bars, one per class
      confusion_matrix.png                           row-normalised 19x19 confusion
      seqXX_frameNNNNNN.png (every --viz-every)      RGB | OneFormer GT | fine
          prediction | coarse prediction

The metric definitions live in SemanticMetricAccumulator so the world-model
evaluator can score decoded rollouts with exactly the same code.
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch

from jepa_drive_wm.data.data_interface_dense import KITTISemanticDataset
from jepa_drive_wm.data.splits import SPLIT_V1
from jepa_drive_wm.evals.common import (
    DEVICE,
    SEMANTICS_CHECKPOINT,
    SEQUENTIAL_CMAP,
    SERIES,
    describe_checkpoint,
    finish_figure,
    load_semantic_decoder,
    markdown_table,
    new_figure,
    style_axes,
    write_metrics_json,
)
from jepa_drive_wm.models.dense_decoders.semantic_decoder import SemanticDecoder
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.train.train_semantics import IGNORE_INDEX, NUM_CLASSES, predict
from jepa_drive_wm.viz.visualiser import (
    CLASS_NAMES,
    CLASS_TO_GROUP,
    GROUP_NAMES,
    NUM_GROUPS,
    class_colors,
    group_colors,
)

FIGURES_DIR = OUTPUTS_DIR / "evals_semantics"
TEST_SEQUENCES = list(SPLIT_V1.test_sequences)

# Binary class subsets used by the coarse metrics, derived from the shared
# 19 -> 5 planning taxonomy in viz.visualiser. Traffic participants are the
# "dynamic object" group: person, rider, car, truck, bus, train, motorcycle,
# bicycle.
DRIVABLE_CLASSES = CLASS_TO_GROUP == GROUP_NAMES.index("drivable")
TRAFFIC_PARTICIPANT_CLASSES = CLASS_TO_GROUP == GROUP_NAMES.index("dynamic object")
SKY_GROUP = GROUP_NAMES.index("sky / ignore")

# Class id (0..18) -> is drivable. Sized 256 so IGNORE_INDEX maps to background.
DRIVABLE_LUT = np.zeros(256, dtype=bool)
DRIVABLE_LUT[:NUM_CLASSES] = DRIVABLE_CLASSES


def _boundary_band(mask: np.ndarray, dilation: int) -> np.ndarray:
    """
    Inner band of a binary mask within `dilation` px of its contour -- the mask
    minus its erosion, as in Boundary IoU [https://arxiv.org/pdf/2103.16562].
    The chessboard distance transform equals `dilation` iterations of the
    reference implementation's 3x3 erosion; padding first makes the image
    border count as contour, also matching the reference.
    """
    padded = np.pad(mask, 1)
    distance = scipy.ndimage.distance_transform_cdt(padded, metric="chessboard")
    return mask & (distance[1:-1, 1:-1] <= dilation)


class SemanticMetricAccumulator:
    """
    Streaming segmentation metrics against OneFormer pseudolabels.

    update() is fed one frame at a time (GT ids, predicted ids) and keeps one
    19x19 confusion matrix plus the drivable boundary counts. IoU convention
    throughout: intersection / union over all labeled pixels fed in
    (dataset-level, not mean-of-frames); classes absent from both GT and
    prediction give NaN.
    """

    def __init__(self, boundary_dilation_ratio: float = 0.02, with_boundary: bool = True) -> None:
        # Boundary band width as a fraction of the image diagonal (paper default).
        self.boundary_dilation_ratio = boundary_dilation_ratio
        self.with_boundary = with_boundary
        self.confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        self.boundary_intersection = 0
        self.boundary_union = 0
        self.frames = 0

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        """gt, pred: (H, W) Cityscapes ids; gt may contain IGNORE_INDEX."""
        gt = gt.astype(np.int64, copy=False)
        pred = pred.astype(np.int64, copy=False)
        labeled = gt != IGNORE_INDEX

        # confusion[i, j] = pixels of GT class i predicted as class j
        self.confusion += np.bincount(
            gt[labeled] * NUM_CLASSES + pred[labeled],
            minlength=NUM_CLASSES ** 2,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        if self.with_boundary:
            dilation = max(1, round(self.boundary_dilation_ratio * math.hypot(*gt.shape)))
            gt_band = _boundary_band(DRIVABLE_LUT[gt], dilation)
            pred_band = _boundary_band(DRIVABLE_LUT[pred], dilation)
            self.boundary_intersection += int((gt_band & pred_band & labeled).sum())
            self.boundary_union += int(((gt_band | pred_band) & labeled).sum())
        self.frames += 1

    # ------------------------------------------------------------------ derived

    @staticmethod
    def _iou_per_row(confusion: np.ndarray) -> np.ndarray:
        """Per-label IoU of a square confusion matrix; NaN where the label never occurs."""
        true_positive = np.diag(confusion)
        union = confusion.sum(axis=0) + confusion.sum(axis=1) - true_positive
        return np.where(union > 0, true_positive / np.maximum(union, 1), np.nan)

    def per_class_iou(self) -> np.ndarray:
        """(19,) IoU per Cityscapes class; NaN where the class never occurs."""
        return self._iou_per_row(self.confusion)

    def group_confusion(self) -> np.ndarray:
        """(5, 5) confusion over the coarse planning groups, collapsed from the fine matrix."""
        group_confusion = np.zeros((NUM_GROUPS, NUM_GROUPS), dtype=np.int64)
        np.add.at(group_confusion, (CLASS_TO_GROUP[:, None], CLASS_TO_GROUP[None, :]), self.confusion)
        return group_confusion

    def _binary_group_iou(self, group_classes: np.ndarray) -> float:
        """
        Binary IoU of a class subset vs everything else, from the fine confusion
        matrix: any within-subset confusion (e.g. car predicted as truck) counts
        as correct.
        """
        on = group_classes
        true_positive = self.confusion[np.ix_(on, on)].sum()
        false_negative = self.confusion[np.ix_(on, ~on)].sum()
        false_positive = self.confusion[np.ix_(~on, on)].sum()
        union = true_positive + false_negative + false_positive
        return float(true_positive / union) if union else float("nan")

    def class_iou(self, cityscapes_class: int | str) -> float:
        if isinstance(cityscapes_class, str):
            cityscapes_class = CLASS_NAMES.index(cityscapes_class)
        return float(self.per_class_iou()[cityscapes_class])

    def mean_iou(self) -> float:
        """Mean IoU over the 19 Cityscapes classes (absent classes excluded)."""
        return float(np.nanmean(self.per_class_iou()))

    def pixel_accuracy(self) -> float:
        total = self.confusion.sum()
        return float(np.trace(self.confusion) / total) if total else float("nan")

    def planning_group_miou(self) -> float:
        """
        Mean IoU over the coarse planning groups (drivable, soft-drivable,
        static obstacle, dynamic object) on the group-collapsed confusion. The
        sky/ignore group's own IoU is discarded, but its pixels still count
        against the other groups' unions.
        """
        per_group = self._iou_per_row(self.group_confusion())
        return float(np.nanmean(np.delete(per_group, SKY_GROUP)))

    def drivable_iou(self) -> float:
        return self._binary_group_iou(DRIVABLE_CLASSES)

    def drivable_boundary_iou(self) -> float:
        if not self.with_boundary or not self.boundary_union:
            return float("nan")
        return self.boundary_intersection / self.boundary_union

    def traffic_participant_iou(self) -> float:
        return self._binary_group_iou(TRAFFIC_PARTICIPANT_CLASSES)

    def summary(self) -> dict[str, float]:
        """The headline numbers in one dict (what the console prints)."""
        return {
            "miou": self.mean_iou(),
            "pixel_accuracy": self.pixel_accuracy(),
            "planning_group_miou": self.planning_group_miou(),
            "drivable_iou": self.drivable_iou(),
            "drivable_boundary_iou": self.drivable_boundary_iou(),
            "traffic_participant_iou": self.traffic_participant_iou(),
            "car_iou": self.class_iou("car"),
            "labeled_pixels": int(self.confusion.sum()),
        }


class ModelPredictions:
    """
    The trained decoder's class-id maps over a dataset, indexable like a list:
    predictions[i] is the (H, W) uint8 prediction for dataset[i].

    Lazy and uncached -- each access is one forward pass -- so a metrics pass
    streams over ~4k frames instead of holding every full-resolution map in
    memory, and SemanticsEvaluator stays decoupled from the model: it accepts
    anything indexable that is frame-aligned with its test set.
    """

    def __init__(self, model: SemanticDecoder, dataset: KITTISemanticDataset) -> None:
        self.model = model
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, item: int) -> np.ndarray:
        sample = self.dataset[item]
        logits = predict(self.model, sample["features"][None].to(DEVICE), sample["target"].shape)
        return logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()


class SemanticsEvaluator:
    """
    Given the test set and frame-aligned predictions of it (from the trained
    model), compute segmentation metrics and qualitative figures.

    One streaming pass over the test set (run lazily on the first metric call
    and cached) feeds a dataset-level SemanticMetricAccumulator and one per
    sequence. `max_frames` caps the pass for smoke tests.
    """

    def __init__(self, test_set: KITTISemanticDataset, predicted_test_set,
                 boundary_dilation_ratio: float = 0.02, figures_dir: Path = FIGURES_DIR,
                 max_frames: int | None = None) -> None:
        self.test_set = test_set
        self.predictions = predicted_test_set
        self.boundary_dilation_ratio = boundary_dilation_ratio
        self.figures_dir = figures_dir
        self.max_frames = max_frames
        self._overall: SemanticMetricAccumulator | None = None
        self._per_sequence: dict[int, SemanticMetricAccumulator] = {}

    # ------------------------------------------------------------------ accumulation

    def _items(self) -> list[int]:
        n = len(self.test_set) if self.max_frames is None else min(self.max_frames, len(self.test_set))
        return list(range(n))

    def _accumulate(self) -> SemanticMetricAccumulator:
        """One pass over the test set: confusion matrix + drivable boundary counts."""
        if self._overall is not None:
            return self._overall

        overall = SemanticMetricAccumulator(self.boundary_dilation_ratio)
        per_sequence: dict[int, SemanticMetricAccumulator] = {}
        for item in self._items():
            sequence_nr, frame_index = self.test_set.index[item]
            gt = self.test_set.sequences[sequence_nr].get_semantics(frame_index)  # (H, W) ids
            pred = self.predictions[item]                                          # (H, W) ids
            overall.update(gt, pred)
            per_sequence.setdefault(
                sequence_nr, SemanticMetricAccumulator(self.boundary_dilation_ratio)
            ).update(gt, pred)

        self._overall, self._per_sequence = overall, per_sequence
        return overall

    # ------------------------------------------------------------------ metrics

    def get_IOU(self, cityscapes_class: int | str) -> float:
        """IoU for one Cityscapes class, by train id (13) or name ("car")."""
        return self._accumulate().class_iou(cityscapes_class)

    def calculate_mean_IOU(self) -> float:
        """Mean IoU over the 19 Cityscapes classes (absent classes excluded)."""
        return self._accumulate().mean_iou()

    def calculate_pixel_accuracy(self) -> float:
        return self._accumulate().pixel_accuracy()

    def calculate_planning_group_mIOU(self) -> float:
        """
        Primary planning-oriented summary: mean IoU over the coarse planning
        groups (drivable, soft-drivable, static obstacle, dynamic object).
        """
        return self._accumulate().planning_group_miou()

    def calculate_drivable_IOU(self) -> float:
        """Binary IoU of the drivable planning group (currently just "road") vs everything else."""
        return self._accumulate().drivable_iou()

    def calculate_drivable_boundary_IOU(self) -> float:
        """
        Boundary IoU [https://arxiv.org/pdf/2103.16562] of the drivable mask:
        plain IoU restricted to a thin band around each mask's contour, so it
        scores edge quality that region IoU washes out.
        """
        return self._accumulate().drivable_boundary_iou()

    def calculate_traffic_participant_IOU(self) -> float:
        """
        Binary IoU of all traffic participants collapsed into one mask (person,
        rider, car, truck, bus, train, motorcycle, bicycle). Within-group
        confusion (car predicted as truck) counts as correct: the question is
        whether the representation preserved the location and extent of traffic
        participants, not their identity. Report alongside get_IOU("car"), where
        identity does matter and KITTI has enough pixels to be meaningful.
        """
        return self._accumulate().traffic_participant_iou()

    def per_sequence_metrics(self) -> dict[int, dict[str, float]]:
        self._accumulate()
        return {seq: acc.summary() for seq, acc in sorted(self._per_sequence.items())}

    def all_metrics(self) -> dict:
        """Everything, ready for metrics.json."""
        overall = self._accumulate()
        per_class = overall.per_class_iou()
        gt_share = overall.confusion.sum(axis=1) / max(overall.confusion.sum(), 1)
        return {
            "frames": overall.frames,
            "overall": overall.summary(),
            "per_class_iou": {name: float(iou) for name, iou in zip(CLASS_NAMES, per_class)},
            "per_class_gt_pixel_share": {name: float(s) for name, s in zip(CLASS_NAMES, gt_share)},
            "per_sequence": {f"{seq:02d}": m for seq, m in self.per_sequence_metrics().items()},
            "confusion": overall.confusion.tolist(),
        }

    # ------------------------------------------------------------------ qualitative

    def plot_per_class_iou(self, path: Path | None = None) -> Path:
        """Horizontal bars: IoU per class (absent classes marked), GT pixel share as label."""
        overall = self._accumulate()
        per_class = overall.per_class_iou()
        share = overall.confusion.sum(axis=1) / max(overall.confusion.sum(), 1)
        y = np.arange(NUM_CLASSES)[::-1]
        fig, axes = new_figure(width=7.0, height=6.5)
        ax = axes[0, 0]
        values = np.nan_to_num(per_class, nan=0.0)
        ax.barh(y, values, height=0.7, color=SERIES[0], linewidth=0)
        for yi, value, s, raw in zip(y, values, share, per_class):
            label = "absent" if np.isnan(raw) else f"{raw:.3f}  ({s:.1%} of GT px)"
            ax.text(value + 0.01, yi, label, va="center", fontsize=7.5, color="#52514e")
        ax.set_yticks(y, CLASS_NAMES, fontsize=8.5)
        ax.set_xlim(0, 1.18)
        style_axes(ax, title=f"Per-class IoU (mIoU {overall.mean_iou():.3f})", xlabel="IoU")
        ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.yaxis.grid(False)
        path = path or self.figures_dir / "per_class_iou.png"
        finish_figure(fig, path)
        return path

    def plot_confusion_matrix(self, path: Path | None = None) -> Path:
        """Row-normalised confusion (GT class -> predicted class), sequential blue ramp."""
        confusion = self._accumulate().confusion.astype(np.float64)
        row_sum = confusion.sum(axis=1, keepdims=True)
        normalised = np.divide(confusion, row_sum, out=np.zeros_like(confusion), where=row_sum > 0)
        fig, axes = new_figure(width=8.5, height=7.5)
        ax = axes[0, 0]
        image = ax.imshow(normalised, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=90, fontsize=8)
        ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=8)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v = normalised[i, j]
                if v >= 0.05:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v > 0.6 else "#0b0b0b")
        ax.set_title("Confusion (rows = OneFormer GT, normalised per row)", loc="left", fontsize=11)
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth")
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="fraction of GT class")
        path = path or self.figures_dir / "confusion_matrix.png"
        finish_figure(fig, path)
        return path

    def visualise_semantic_segmentation(self, viz_every: int = 100) -> None:
        """Every `viz_every` frames: RGB, OneFormer GT, fine and coarse predictions in one PNG."""
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        for item in self._items()[::viz_every]:
            sequence_nr, frame_index = self.test_set.index[item]
            sequence = self.test_set.sequences[sequence_nr]
            pred = self.predictions[item]

            panels = [
                (sequence.get_image(frame_index), f"seq {sequence_nr:02d} frame {frame_index}"),
                (class_colors(sequence.get_semantics(frame_index)), "OneFormer semantics"),
                (class_colors(pred), "predicted (fine)"),
                (group_colors(pred), "predicted (coarse)"),
            ]
            fig, axes = plt.subplots(len(panels), 1, figsize=(10, 12))
            for ax, (panel, title) in zip(axes, panels):
                ax.imshow(panel)
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(self.figures_dir / f"seq{sequence_nr:02d}_frame{frame_index:06d}.png", dpi=150)
            plt.close(fig)


# ----------------------------------------------------------------------------- reporting

HEADLINE = [
    ("miou", "mean IoU (19-class)"),
    ("pixel_accuracy", "pixel accuracy"),
    ("planning_group_miou", "planning-group mIoU (no sky)"),
    ("drivable_iou", "drivable IoU"),
    ("drivable_boundary_iou", "drivable boundary IoU"),
    ("traffic_participant_iou", "traffic-participant IoU"),
    ("car_iou", "car IoU"),
]


def metrics_markdown(metrics: dict, checkpoint_line: str) -> str:
    overall = metrics["overall"]
    headline_rows = [[label, overall[key]] for key, label in HEADLINE]
    class_rows = [[name, metrics["per_class_iou"][name], metrics["per_class_gt_pixel_share"][name]]
                  for name in CLASS_NAMES]
    seq_rows = [[seq, m["miou"], m["planning_group_miou"], m["drivable_iou"],
                 m["drivable_boundary_iou"], m["traffic_participant_iou"], m["car_iou"]]
                for seq, m in metrics["per_sequence"].items()]
    return "\n".join([
        "# Semantic decoder -- test set (SPLIT_V1: sequences "
        + ", ".join(f"{s:02d}" for s in TEST_SEQUENCES) + ")",
        "",
        f"checkpoint: `{checkpoint_line}`  ",
        f"frames: {metrics['frames']}; IoU = intersection/union pooled over the test set "
        "(dataset-level, not mean-of-frames)",
        "",
        "## Headline",
        "",
        markdown_table(["metric", "value"], headline_rows),
        "",
        "## Per class",
        "",
        markdown_table(["class", "IoU", "GT pixel share"], class_rows),
        "",
        "## Per sequence",
        "",
        markdown_table(["seq", "mIoU", "planning mIoU", "drivable IoU", "drivable bIoU",
                        "traffic IoU", "car IoU"], seq_rows),
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained semantic decoder on the test split")
    parser.add_argument("--checkpoint", type=Path, default=SEMANTICS_CHECKPOINT)
    parser.add_argument("--viz-every", type=int, default=100,
                        help="save a figure every N test frames (0 disables)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="only score the first N test frames (smoke tests)")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    model, checkpoint = load_semantic_decoder(args.checkpoint)
    checkpoint_line = describe_checkpoint(args.checkpoint, checkpoint)
    print(f"loaded {checkpoint_line}")

    dataset = KITTISemanticDataset(TEST_SEQUENCES)
    evaluator = SemanticsEvaluator(dataset, ModelPredictions(model, dataset),
                                   figures_dir=args.figures_dir, max_frames=args.max_frames)

    print("\nper-class IoU:")
    for class_name in CLASS_NAMES:
        iou = evaluator.get_IOU(class_name)
        print(f"  {class_name:<14} " + ("absent" if math.isnan(iou) else f"{iou:.4f}"))
    print(f"\nmean IoU (19-class)           {evaluator.calculate_mean_IOU():.4f}")
    print(f"pixel accuracy                {evaluator.calculate_pixel_accuracy():.4f}")
    print(f"planning-group mIoU (no sky)  {evaluator.calculate_planning_group_mIOU():.4f}")
    print(f"drivable IoU                  {evaluator.calculate_drivable_IOU():.4f}")
    print(f"drivable boundary IoU         {evaluator.calculate_drivable_boundary_IOU():.4f}")
    print(f"traffic-participant IoU       {evaluator.calculate_traffic_participant_IOU():.4f}")
    print(f"car IoU                       {evaluator.get_IOU('car'):.4f}")

    metrics = evaluator.all_metrics()
    metrics["checkpoint"] = checkpoint_line
    metrics["test_sequences"] = TEST_SEQUENCES
    print("\nper sequence (mIoU / planning mIoU / drivable IoU / traffic IoU):")
    for seq, m in metrics["per_sequence"].items():
        print(f"  seq {seq}: {m['miou']:.4f} / {m['planning_group_miou']:.4f} / "
              f"{m['drivable_iou']:.4f} / {m['traffic_participant_iou']:.4f}")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(metrics, args.figures_dir / "metrics.json")
    (args.figures_dir / "metrics.md").write_text(metrics_markdown(metrics, checkpoint_line))
    evaluator.plot_per_class_iou()
    evaluator.plot_confusion_matrix()
    if args.viz_every:
        evaluator.visualise_semantic_segmentation(args.viz_every)
    print(f"\nmetrics + figures saved to {args.figures_dir}")


if __name__ == "__main__":
    main()
