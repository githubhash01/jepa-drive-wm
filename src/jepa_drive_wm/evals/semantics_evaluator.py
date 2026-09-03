"""
Evaluate the trained semantic decoder on the held-out KITTI test sequences.

The evaluator uses dataset-level pooling: one 19 x 19 confusion matrix is
accumulated over all labelled pixels in the evaluated frames, and the metrics
are computed from those pooled counts. The same metrics are also reported for
each held-out sequence.

Quantitative outputs in OUTPUTS_DIR/evals_semantics:

    metrics.json
        Complete machine-readable results.

    metrics.md
        Thesis-facing tables containing the headline metrics, per-sequence
        results, and per-class IoU with OneFormer pseudo-label support.

Qualitative outputs:

    semantic_decoder_examples.png
        One representative frame per held-out sequence, read from top to
        bottom: RGB image, OneFormer fine pseudo-labels, and semantics decoded
        from the V-JEPA representation.

    semantic_decoder_planning_examples.png
        The equivalent comparison after grouping the 19 Cityscapes classes
        into the coarser planning categories used by this project.

    per_class_iou.png
        Fine-class IoU with the OneFormer pseudo-label pixel share shown for
        each class.

    confusion_matrix.png
        Row-normalised fine-class confusion matrix.

By default, the representative frame for each sequence is the frame whose
planning-group mIoU is closest to that sequence's median frame score. This is
used only to select qualitative examples; all reported quantitative metrics
remain dataset-level pooled metrics.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    load_semantic_decoder,
    markdown_table,
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

# Planning-oriented binary class subsets. Traffic participants are all classes
# in the project's "dynamic object" group.
DRIVABLE_CLASSES = CLASS_TO_GROUP == GROUP_NAMES.index("drivable")
TRAFFIC_PARTICIPANT_CLASSES = CLASS_TO_GROUP == GROUP_NAMES.index("dynamic object")
SKY_GROUP = GROUP_NAMES.index("sky / ignore")
SKY_CLASS = CLASS_NAMES.index("sky")

# Class id -> is drivable. The table has 256 entries so OneFormer's ignored
# value 255 maps safely to False.
DRIVABLE_LUT = np.zeros(256, dtype=bool)
DRIVABLE_LUT[:NUM_CLASSES] = DRIVABLE_CLASSES

# LaTeX-like thesis typography without requiring a system LaTeX installation.
# text.usetex remains False so the evaluator is portable across machines.
matplotlib.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": [
            "CMU Serif",
            "Computer Modern Roman",
            "STIX Two Text",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
    }
)


# ----------------------------------------------------------------------------- helpers


def _format_percent(fraction: float) -> str:
    """Format small supports without incorrectly displaying them as 0%."""
    if not math.isfinite(fraction):
        return "n/a"
    percentage = 100.0 * fraction
    if percentage == 0:
        return "0%"
    if percentage < 0.01:
        return "<0.01%"
    if percentage < 0.1:
        return f"{percentage:.2f}%"
    return f"{percentage:.1f}%"


def _boundary_band(mask: np.ndarray, dilation: int) -> np.ndarray:
    """
    Inner band of a binary mask within ``dilation`` pixels of its contour.

    Padding first makes the image border count as part of the contour, matching
    the convention used in the original evaluator.
    """
    padded = np.pad(mask, 1)
    distance = scipy.ndimage.distance_transform_cdt(padded, metric="chessboard")
    return mask & (distance[1:-1, 1:-1] <= dilation)


def _normalise_colour(pixel: np.ndarray) -> tuple[float, float, float]:
    """Convert one RGB colour returned by the visualiser to Matplotlib RGB."""
    colour = np.asarray(pixel, dtype=np.float64).reshape(-1)[:3]
    if colour.max(initial=0.0) > 1.0:
        colour = colour / 255.0
    return tuple(float(value) for value in colour)


def _fine_colour_image(labels: np.ndarray) -> np.ndarray:
    """Render fine labels, showing OneFormer-unlabelled pixels in neutral grey."""
    labels = labels.astype(np.int64, copy=False)
    labelled = (labels >= 0) & (labels < NUM_CLASSES)
    safe = np.where(labelled, labels, 0).astype(np.uint8)
    image = np.asarray(class_colors(safe)).copy()

    if image.dtype.kind in "ui":
        ignore_colour = np.array([150, 150, 150], dtype=image.dtype)
    else:
        ignore_colour = np.array([0.59, 0.59, 0.59], dtype=image.dtype)
    image[~labelled] = ignore_colour
    return image


def _planning_colour_image(labels: np.ndarray) -> np.ndarray:
    """Render the project's coarse planning groups from fine Cityscapes ids."""
    labels = labels.astype(np.int64, copy=False)
    labelled = (labels >= 0) & (labels < NUM_CLASSES)
    # The project's final group is explicitly named sky / ignore, so unlabelled
    # target pixels are mapped to the sky class for this qualitative view.
    safe = np.where(labelled, labels, SKY_CLASS).astype(np.uint8)
    return np.asarray(group_colors(safe))


def _fine_legend_handles() -> list[Patch]:
    handles = []
    for class_id, name in enumerate(CLASS_NAMES):
        sample = class_colors(np.array([[class_id]], dtype=np.uint8))[0, 0]
        handles.append(Patch(facecolor=_normalise_colour(sample), edgecolor="none", label=name))
    handles.append(Patch(facecolor=(0.59, 0.59, 0.59), edgecolor="none", label="unlabelled"))
    return handles


def _planning_legend_handles() -> list[Patch]:
    handles = []
    for group_id, name in enumerate(GROUP_NAMES):
        members = np.flatnonzero(CLASS_TO_GROUP == group_id)
        if members.size == 0:
            continue
        sample = group_colors(np.array([[members[0]]], dtype=np.uint8))[0, 0]
        handles.append(Patch(facecolor=_normalise_colour(sample), edgecolor="none", label=name))
    return handles


# ----------------------------------------------------------------------------- metrics


class SemanticMetricAccumulator:
    """
    Streaming semantic metrics against OneFormer pseudo-labels.

    ``update`` receives one target and one prediction map at a time. It adds
    their counts to one 19 x 19 confusion matrix. IoUs are calculated from the
    pooled confusion matrix, not separately per frame. Classes absent from both
    target and prediction have undefined IoU and are excluded from mIoU.

    This public interface is intentionally retained because the world-model
    evaluators import this class to evaluate decoded predicted latents.
    """

    def __init__(self, boundary_dilation_ratio: float = 0.02, with_boundary: bool = True) -> None:
        self.boundary_dilation_ratio = boundary_dilation_ratio
        self.with_boundary = with_boundary
        self.confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        self.boundary_intersection = 0
        self.boundary_union = 0
        self.frames = 0

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        """Update from one frame; ``gt`` may contain ``IGNORE_INDEX``."""
        gt = gt.astype(np.int64, copy=False)
        pred = pred.astype(np.int64, copy=False)
        labelled = gt != IGNORE_INDEX

        # confusion[i, j] = target class i predicted as class j
        self.confusion += np.bincount(
            gt[labelled] * NUM_CLASSES + pred[labelled],
            minlength=NUM_CLASSES**2,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        if self.with_boundary:
            dilation = max(1, round(self.boundary_dilation_ratio * math.hypot(*gt.shape)))
            gt_band = _boundary_band(DRIVABLE_LUT[gt], dilation)
            pred_band = _boundary_band(DRIVABLE_LUT[pred], dilation)
            self.boundary_intersection += int((gt_band & pred_band & labelled).sum())
            self.boundary_union += int(((gt_band | pred_band) & labelled).sum())
        self.frames += 1

    @staticmethod
    def _iou_per_row(confusion: np.ndarray) -> np.ndarray:
        true_positive = np.diag(confusion).astype(np.float64)
        union = confusion.sum(axis=0) + confusion.sum(axis=1) - true_positive
        result = np.full(confusion.shape[0], np.nan, dtype=np.float64)
        np.divide(true_positive, union, out=result, where=union > 0)
        return result

    def per_class_iou(self) -> np.ndarray:
        return self._iou_per_row(self.confusion)

    def group_confusion(self) -> np.ndarray:
        group_confusion = np.zeros((NUM_GROUPS, NUM_GROUPS), dtype=np.int64)
        np.add.at(
            group_confusion,
            (CLASS_TO_GROUP[:, None], CLASS_TO_GROUP[None, :]),
            self.confusion,
        )
        return group_confusion

    def _binary_group_iou(self, group_classes: np.ndarray) -> float:
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
        return float(np.nanmean(self.per_class_iou()))

    def pixel_accuracy(self) -> float:
        total = self.confusion.sum()
        return float(np.trace(self.confusion) / total) if total else float("nan")

    def planning_group_miou(self) -> float:
        """
        Mean IoU over drivable, soft-drivable, static-obstacle and dynamic-
        object groups. The sky / ignore group's own IoU is excluded, although
        sky pixels can still create false positives for another group.
        """
        per_group = self._iou_per_row(self.group_confusion())
        return float(np.nanmean(np.delete(per_group, SKY_GROUP)))

    def drivable_iou(self) -> float:
        return self._binary_group_iou(DRIVABLE_CLASSES)

    def drivable_boundary_iou(self) -> float:
        if not self.with_boundary or not self.boundary_union:
            return float("nan")
        return float(self.boundary_intersection / self.boundary_union)

    def traffic_participant_iou(self) -> float:
        return self._binary_group_iou(TRAFFIC_PARTICIPANT_CLASSES)

    def summary(self) -> dict[str, float | int]:
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
    """Lazy, frame-aligned class predictions from the trained decoder."""

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
    """Compute dataset-level metrics and thesis-ready semantic figures."""

    def __init__(
        self,
        test_set: KITTISemanticDataset,
        predicted_test_set,
        boundary_dilation_ratio: float = 0.02,
        figures_dir: Path = FIGURES_DIR,
        max_frames: int | None = None,
    ) -> None:
        self.test_set = test_set
        self.predictions = predicted_test_set
        self.boundary_dilation_ratio = boundary_dilation_ratio
        self.figures_dir = figures_dir
        self.max_frames = max_frames
        self._overall: SemanticMetricAccumulator | None = None
        self._per_sequence: dict[int, SemanticMetricAccumulator] = {}
        self._frame_records: list[dict] = []

    def _items(self) -> list[int]:
        count = len(self.test_set) if self.max_frames is None else min(self.max_frames, len(self.test_set))
        return list(range(count))

    def _accumulate(self) -> SemanticMetricAccumulator:
        """One streaming pass over the evaluated frames."""
        if self._overall is not None:
            return self._overall

        overall = SemanticMetricAccumulator(self.boundary_dilation_ratio)
        per_sequence: dict[int, SemanticMetricAccumulator] = {}
        frame_records: list[dict] = []

        for item in self._items():
            sequence_nr, frame_index = self.test_set.index[item]
            gt = self.test_set.sequences[sequence_nr].get_semantics(frame_index)
            pred = self.predictions[item]

            overall.update(gt, pred)
            per_sequence.setdefault(
                sequence_nr,
                SemanticMetricAccumulator(self.boundary_dilation_ratio),
            ).update(gt, pred)

            # Frame-level scores are stored only for reproducible qualitative
            # example selection. They are not used for the headline metrics.
            frame_accumulator = SemanticMetricAccumulator(with_boundary=False)
            frame_accumulator.update(gt, pred)
            frame_summary = frame_accumulator.summary()
            frame_records.append(
                {
                    "item": int(item),
                    "sequence": int(sequence_nr),
                    "frame": int(frame_index),
                    "miou": float(frame_summary["miou"]),
                    "planning_group_miou": float(frame_summary["planning_group_miou"]),
                }
            )

        self._overall = overall
        self._per_sequence = per_sequence
        self._frame_records = frame_records
        return overall

    # ---------------------------------------------------------------- metrics

    def get_IOU(self, cityscapes_class: int | str) -> float:
        return self._accumulate().class_iou(cityscapes_class)

    def calculate_mean_IOU(self) -> float:
        return self._accumulate().mean_iou()

    def calculate_pixel_accuracy(self) -> float:
        return self._accumulate().pixel_accuracy()

    def calculate_planning_group_mIOU(self) -> float:
        return self._accumulate().planning_group_miou()

    def calculate_drivable_IOU(self) -> float:
        return self._accumulate().drivable_iou()

    def calculate_drivable_boundary_IOU(self) -> float:
        return self._accumulate().drivable_boundary_iou()

    def calculate_traffic_participant_IOU(self) -> float:
        return self._accumulate().traffic_participant_iou()

    def per_sequence_metrics(self) -> dict[int, dict[str, float | int]]:
        self._accumulate()
        return {
            sequence: accumulator.summary()
            for sequence, accumulator in sorted(self._per_sequence.items())
        }

    def all_metrics(self) -> dict:
        overall = self._accumulate()
        per_class = overall.per_class_iou()
        class_pixels = overall.confusion.sum(axis=1)
        total_pixels = max(int(class_pixels.sum()), 1)
        class_share = class_pixels / total_pixels

        return {
            "frames": overall.frames,
            "aggregation": "dataset-level confusion matrix pooled over all labelled pixels",
            "test_sequences": TEST_SEQUENCES,
            "overall": overall.summary(),
            "per_class_iou": {
                name: float(iou) for name, iou in zip(CLASS_NAMES, per_class)
            },
            "per_class_pseudolabel_pixels": {
                name: int(pixels) for name, pixels in zip(CLASS_NAMES, class_pixels)
            },
            "per_class_pseudolabel_pixel_share": {
                name: float(share) for name, share in zip(CLASS_NAMES, class_share)
            },
            # Retain the old key for compatibility with any downstream notes.
            "per_class_gt_pixel_share": {
                name: float(share) for name, share in zip(CLASS_NAMES, class_share)
            },
            "per_sequence": {
                f"{sequence:02d}": values
                for sequence, values in self.per_sequence_metrics().items()
            },
            "confusion": overall.confusion.tolist(),
        }

    # ------------------------------------------------------- example selection

    def select_example_records(self, selection: str = "median") -> list[dict]:
        """
        Select one reproducible example per held-out sequence.

        median: frame nearest the sequence median planning-group mIoU
        worst:  frame with the lowest planning-group mIoU
        middle: temporal middle frame among the evaluated frames
        """
        if selection not in {"median", "worst", "middle"}:
            raise ValueError(f"unknown example selection: {selection!r}")

        self._accumulate()
        selected: list[dict] = []
        for sequence in TEST_SEQUENCES:
            records = [
                record
                for record in self._frame_records
                if record["sequence"] == sequence
                and math.isfinite(record["planning_group_miou"])
            ]
            if not records:
                continue

            if selection == "worst":
                chosen = min(records, key=lambda record: record["planning_group_miou"])
            elif selection == "middle":
                chosen = records[len(records) // 2]
            else:
                median_score = float(
                    np.median([record["planning_group_miou"] for record in records])
                )
                chosen = min(
                    records,
                    key=lambda record: abs(record["planning_group_miou"] - median_score),
                )
            selected.append(chosen)
        return selected

    # ---------------------------------------------------------------- figures

    def plot_per_class_iou(self, path: Path | None = None) -> Path:
        """Fine-class IoU with pseudo-label support shown beside each bar."""
        overall = self._accumulate()
        per_class = overall.per_class_iou()
        class_pixels = overall.confusion.sum(axis=1)
        share = class_pixels / max(int(class_pixels.sum()), 1)

        y = np.arange(NUM_CLASSES)
        values = np.nan_to_num(per_class, nan=0.0)

        fig, ax = plt.subplots(figsize=(7.2, 6.5))
        ax.barh(y, values, height=0.68, color=SERIES[0], linewidth=0)
        ax.set_yticks(y, CLASS_NAMES)
        ax.invert_yaxis()

        for position, value, support, raw in zip(y, values, share, per_class):
            if np.isnan(raw):
                label = "absent"
            else:
                label = f"IoU {raw:.3f}; {_format_percent(float(support))} of pseudo-label pixels"
            ax.text(value + 0.012, position, label, va="center", fontsize=7.5)

        ax.set_xlim(0, 1.34)
        style_axes(
            ax,
            title="Fine-class semantic decoding",
            xlabel="Intersection over union",
        )
        ax.xaxis.grid(True, alpha=0.65)
        ax.yaxis.grid(False)
        ax.text(
            0.0,
            1.015,
            f"Dataset-level mIoU = {overall.mean_iou():.3f}; support from OneFormer pseudo-labels",
            transform=ax.transAxes,
            fontsize=8.5,
            va="bottom",
        )

        path = path or self.figures_dir / "per_class_iou.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_confusion_matrix(self, path: Path | None = None) -> Path:
        """Row-normalised fine-class confusion matrix."""
        confusion = self._accumulate().confusion.astype(np.float64)
        row_sum = confusion.sum(axis=1, keepdims=True)
        normalised = np.divide(
            confusion,
            row_sum,
            out=np.zeros_like(confusion),
            where=row_sum > 0,
        )

        fig, ax = plt.subplots(figsize=(8.7, 7.7))
        image = ax.imshow(normalised, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=90)
        ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)

        # Annotate only substantial confusions to keep a 19 x 19 matrix legible.
        for row in range(NUM_CLASSES):
            for column in range(NUM_CLASSES):
                value = normalised[row, column]
                if value >= 0.10:
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value > 0.58 else "black",
                    )

        ax.set_title("Fine-class confusion", loc="left", pad=10)
        ax.set_xlabel("Decoded class")
        ax.set_ylabel("OneFormer pseudo-label")
        colour_bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
        colour_bar.set_label("Fraction of each pseudo-label class")

        path = path or self.figures_dir / "confusion_matrix.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_example_grid(
        self,
        *,
        planning: bool,
        selection: str,
        path: Path,
    ) -> Path:
        records = self.select_example_records(selection)
        if not records:
            raise RuntimeError("No semantic frames were available for visualisation")

        columns = len(records)
        figure_height = 5.7 if planning else 6.2
        fig, axes = plt.subplots(
            3,
            columns,
            figsize=(4.15 * columns, figure_height),
            squeeze=False,
        )

        for column, record in enumerate(records):
            item = int(record["item"])
            sequence_nr = int(record["sequence"])
            frame_index = int(record["frame"])
            sequence = self.test_set.sequences[sequence_nr]

            rgb = sequence.get_image(frame_index)
            target = sequence.get_semantics(frame_index)
            prediction = self.predictions[item]

            if planning:
                target_image = _planning_colour_image(target)
                prediction_image = _planning_colour_image(prediction)
                score_name = "planning mIoU"
                score = record["planning_group_miou"]
            else:
                target_image = _fine_colour_image(target)
                prediction_image = _fine_colour_image(prediction)
                score_name = "frame mIoU"
                score = record["miou"]

            axes[0, column].imshow(rgb)
            axes[1, column].imshow(target_image)
            axes[2, column].imshow(prediction_image)
            axes[0, column].set_title(
                f"Sequence {sequence_nr:02d}, frame {frame_index}\n"
                f"{score_name} = {score:.3f}",
                pad=7,
            )

            for row in range(3):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                for spine in axes[row, column].spines.values():
                    spine.set_visible(False)

        if planning:
            row_labels = (
                "RGB image",
                "OneFormer\nplanning groups",
                "Decoded\nplanning groups",
            )
            handles = _planning_legend_handles()
            legend_columns = min(len(handles), 5)
            bottom = 0.12
        else:
            row_labels = (
                "RGB image",
                "OneFormer\npseudo-labels",
                "Decoded V-JEPA\nsemantics",
            )
            handles = _fine_legend_handles()
            legend_columns = 7
            bottom = 0.20

        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(
                label,
                rotation=90,
                va="center",
                labelpad=18,
                fontsize=10,
            )

        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=legend_columns,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            columnspacing=0.9,
            handlelength=1.1,
            handletextpad=0.35,
        )
        fig.subplots_adjust(
            left=0.09,
            right=0.995,
            top=0.91,
            bottom=bottom,
            wspace=0.025,
            hspace=0.08,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_thesis_examples(self, selection: str = "median") -> tuple[Path, Path]:
        """Generate fine-class and planning-group qualitative figures."""
        fine_path = self.figures_dir / "semantic_decoder_examples.png"
        planning_path = self.figures_dir / "semantic_decoder_planning_examples.png"
        self._plot_example_grid(
            planning=False,
            selection=selection,
            path=fine_path,
        )
        self._plot_example_grid(
            planning=True,
            selection=selection,
            path=planning_path,
        )
        return fine_path, planning_path

    def visualise_semantic_segmentation(self, viz_every: int) -> None:
        """
        Optional legacy diagnostic output: one four-panel image every N frames.

        This is disabled by default because the two compact thesis figures are
        usually more useful than hundreds of per-frame files.
        """
        if viz_every <= 0:
            return
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        for item in self._items()[::viz_every]:
            sequence_nr, frame_index = self.test_set.index[item]
            sequence = self.test_set.sequences[sequence_nr]
            target = sequence.get_semantics(frame_index)
            prediction = self.predictions[item]

            panels = [
                (sequence.get_image(frame_index), "RGB image"),
                (_fine_colour_image(target), "OneFormer pseudo-labels"),
                (_fine_colour_image(prediction), "Decoded fine semantics"),
                (_planning_colour_image(prediction), "Decoded planning groups"),
            ]
            fig, axes = plt.subplots(4, 1, figsize=(10, 10))
            for ax, (panel, title) in zip(axes, panels):
                ax.imshow(panel)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(f"Sequence {sequence_nr:02d}, frame {frame_index}")
            fig.tight_layout()
            fig.savefig(
                self.figures_dir / f"seq{sequence_nr:02d}_frame{frame_index:06d}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)


# ----------------------------------------------------------------------------- reporting


HEADLINE = [
    ("miou", "Fine-class mIoU"),
    ("planning_group_miou", "Planning-group mIoU (excluding sky)"),
    ("drivable_iou", "Drivable IoU"),
    ("drivable_boundary_iou", "Drivable boundary IoU"),
    ("traffic_participant_iou", "Traffic-participant IoU"),
    ("car_iou", "Car IoU"),
    ("pixel_accuracy", "Pixel accuracy (supporting)"),
]


def metrics_markdown(metrics: dict, checkpoint_line: str) -> str:
    """Create the thesis-facing Markdown report."""
    overall = metrics["overall"]
    headline_rows = [[label, overall[key]] for key, label in HEADLINE]

    class_rows = [
        [
            name,
            metrics["per_class_iou"][name],
            _format_percent(metrics["per_class_pseudolabel_pixel_share"][name]),
            metrics["per_class_pseudolabel_pixels"][name],
        ]
        for name in CLASS_NAMES
    ]

    sequence_rows = [
        [
            sequence,
            values["miou"],
            values["planning_group_miou"],
            values["drivable_iou"],
            values["drivable_boundary_iou"],
            values["traffic_participant_iou"],
            values["car_iou"],
            values["pixel_accuracy"],
        ]
        for sequence, values in metrics["per_sequence"].items()
    ]

    return "\n".join(
        [
            "# Semantic decoder -- held-out test set",
            "",
            "test sequences: " + ", ".join(f"{sequence:02d}" for sequence in TEST_SEQUENCES) + "  ",
            f"checkpoint: `{checkpoint_line}`  ",
            f"frames: {metrics['frames']}; labelled pixels: {overall['labeled_pixels']:,}  ",
            "aggregation: one confusion matrix pooled over all labelled pixels in the evaluated frames",
            "",
            "## Headline results",
            "",
            markdown_table(["metric", "value"], headline_rows),
            "",
            "Fine-class mIoU gives every Cityscapes class equal weight after dataset-level pooling; "
            "classes absent from both the OneFormer pseudo-labels and the decoder predictions are excluded. "
            "Planning-group mIoU averages drivable, soft-drivable, static-obstacle and dynamic-object IoU; "
            "the sky group's own IoU is excluded. Pixel accuracy is retained as a supporting metric because "
            "large common classes contribute most of its pixels.",
            "",
            "## Results by sequence",
            "",
            markdown_table(
                [
                    "sequence",
                    "mIoU",
                    "planning mIoU",
                    "drivable IoU",
                    "drivable bIoU",
                    "traffic IoU",
                    "car IoU",
                    "pixel accuracy",
                ],
                sequence_rows,
            ),
            "",
            "## Fine-class results",
            "",
            markdown_table(
                ["class", "IoU", "pseudo-label pixel share", "pseudo-label pixels"],
                class_rows,
            ),
            "",
            "Traffic-participant IoU combines person, rider, car, truck, bus, train, motorcycle and bicycle "
            "into one foreground region. Confusions within that group therefore count as spatially correct for "
            "this metric, while car IoU still requires the specific car class.",
            "",
        ]
    )


def print_overview(metrics: dict) -> None:
    overall = metrics["overall"]
    print("\nHeadline semantic-decoder results")
    print(f"{'metric':<40} {'value':>10}")
    for key, label in HEADLINE:
        print(f"{label:<40} {overall[key]:>10.4f}")

    print("\nPer-sequence results")
    print(
        f"{'seq':<6} {'mIoU':>8} {'plan mIoU':>11} {'drive IoU':>11} "
        f"{'drive bIoU':>12} {'traffic':>10} {'car':>8}"
    )
    for sequence, values in metrics["per_sequence"].items():
        print(
            f"{sequence:<6} {values['miou']:>8.4f} {values['planning_group_miou']:>11.4f} "
            f"{values['drivable_iou']:>11.4f} {values['drivable_boundary_iou']:>12.4f} "
            f"{values['traffic_participant_iou']:>10.4f} {values['car_iou']:>8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained semantic decoder")
    parser.add_argument("--checkpoint", type=Path, default=SEMANTICS_CHECKPOINT)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="score only the first N test frames (for smoke tests)",
    )
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--example-selection",
        choices=("median", "worst", "middle"),
        default="median",
        help="select median-performance, worst, or temporal-middle examples per sequence",
    )
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="do not generate the two thesis qualitative figures",
    )
    parser.add_argument(
        "--skip-confusion",
        action="store_true",
        help="do not generate the confusion-matrix figure",
    )
    parser.add_argument(
        "--skip-per-class-plot",
        action="store_true",
        help="do not generate the per-class IoU figure",
    )
    parser.add_argument(
        "--viz-every",
        type=int,
        default=0,
        help="optional legacy diagnostics: save one per-frame figure every N frames; 0 disables",
    )
    args = parser.parse_args()

    model, checkpoint = load_semantic_decoder(args.checkpoint)
    checkpoint_line = describe_checkpoint(args.checkpoint, checkpoint)
    print(f"loaded {checkpoint_line}")

    dataset = KITTISemanticDataset(TEST_SEQUENCES)
    evaluator = SemanticsEvaluator(
        dataset,
        ModelPredictions(model, dataset),
        figures_dir=args.figures_dir,
        max_frames=args.max_frames,
    )

    metrics = evaluator.all_metrics()
    metrics["checkpoint"] = checkpoint_line

    example_records = evaluator.select_example_records(args.example_selection)
    metrics["qualitative_examples"] = {
        "selection": args.example_selection,
        "selection_metric": "per-frame planning-group mIoU",
        "frames": [
            {
                "sequence": record["sequence"],
                "frame": record["frame"],
                "frame_miou": record["miou"],
                "frame_planning_group_miou": record["planning_group_miou"],
            }
            for record in example_records
        ],
    }

    print_overview(metrics)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.figures_dir / "metrics.json"
    report_path = args.figures_dir / "metrics.md"
    write_metrics_json(metrics, metrics_path)
    report_path.write_text(metrics_markdown(metrics, checkpoint_line))

    print(f"\nmetrics: {metrics_path}")
    print(f"report:  {report_path}")

    if not args.skip_per_class_plot:
        path = evaluator.plot_per_class_iou()
        print(f"per-class figure: {path}")

    if not args.skip_confusion:
        path = evaluator.plot_confusion_matrix()
        print(f"confusion figure: {path}")

    if not args.skip_examples:
        fine_path, planning_path = evaluator.plot_thesis_examples(args.example_selection)
        print(f"fine qualitative figure:     {fine_path}")
        print(f"planning qualitative figure: {planning_path}")

    evaluator.visualise_semantic_segmentation(args.viz_every)


if __name__ == "__main__":
    main()
