"""
Evaluate the trained semantic decoder on the held-out KITTI test sequences.

- load the best checkpoint
- report segmentation metrics via SemanticsEvaluator:
      per-class IoU, 19-class mIoU, non-sky planning-group mIoU,
      drivable IoU, drivable Boundary IoU, traffic-participant IoU, car IoU
- save side-by-side (RGB | OneFormer GT | fine prediction | coarse prediction) figures
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch

from jepa_drive_wm.dense_decoder.data_interface_dense import KITTISemanticDataset
from jepa_drive_wm.dense_decoder.model_semantics import SemanticDecoder
from jepa_drive_wm.dense_decoder.train_semantics import (
    CHECKPOINT_PATH,
    DEVICE,
    IGNORE_INDEX,
    NUM_CLASSES,
    TEST_SEQUENCES,
    predict,
)
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.viz.visualiser import (
    CLASS_NAMES,
    CLASS_TO_GROUP,
    GROUP_NAMES,
    NUM_GROUPS,
    class_colors,
    group_colors,
)

FIGURES_DIR = OUTPUTS_DIR / "evals_semantics"

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


class ModelPredictions:
    """
    The trained decoder's class-id maps over a dataset, indexable like a list:
    predictions[i] is the (H, W) uint8 prediction for dataset[i].

    Lazy and uncached -- each access is one forward pass -- so a metrics pass
    streams over ~2k frames instead of holding every full-resolution map in
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


class SemanticsEvaluator:
    """
    Given the test set and frame-aligned predictions of it (from the trained
    model), compute segmentation metrics and qualitative figures.

    All confusion-based metrics come from a single pass over the test set that
    accumulates one 19x19 confusion matrix plus the drivable boundary counts,
    run lazily on the first metric call and cached. IoU convention throughout:
    intersection / union over all labeled pixels of the whole test set
    (dataset-level, not mean-of-frames); classes absent from both GT and
    prediction give NaN.
    """

    def __init__(self, test_set: KITTISemanticDataset, predicted_test_set,
                 boundary_dilation_ratio: float = 0.02) -> None:
        self.test_set = test_set
        self.predictions = predicted_test_set
        # Boundary band width as a fraction of the image diagonal (paper default).
        self.boundary_dilation_ratio = boundary_dilation_ratio
        self.figures_dir = FIGURES_DIR
        self._stats: dict | None = None

    # ------------------------------------------------------------------ accumulation

    def _accumulate(self) -> dict:
        """One pass over the test set: confusion matrix + drivable boundary counts."""
        if self._stats is not None:
            return self._stats

        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        boundary_intersection = 0
        boundary_union = 0

        for item, (sequence_nr, frame_index) in enumerate(self.test_set.index):
            gt = self.test_set.sequences[sequence_nr].get_semantics(frame_index)  # (H, W) ids
            pred = self.predictions[item].astype(np.int64)                        # (H, W) ids
            labeled = gt != IGNORE_INDEX

            # confusion[i, j] = pixels of GT class i predicted as class j
            confusion += np.bincount(
                gt[labeled] * NUM_CLASSES + pred[labeled],
                minlength=NUM_CLASSES ** 2,
            ).reshape(NUM_CLASSES, NUM_CLASSES)

            dilation = max(1, round(self.boundary_dilation_ratio * math.hypot(*gt.shape)))
            gt_band = _boundary_band(DRIVABLE_LUT[gt], dilation)
            pred_band = _boundary_band(DRIVABLE_LUT[pred], dilation)
            boundary_intersection += int((gt_band & pred_band & labeled).sum())
            boundary_union += int(((gt_band | pred_band) & labeled).sum())

        self._stats = {
            "confusion": confusion,
            "boundary_intersection": boundary_intersection,
            "boundary_union": boundary_union,
        }
        return self._stats

    def _per_class_IOU(self) -> np.ndarray:
        """(19,) IoU per Cityscapes class; NaN where the class never occurs."""
        return self._iou_per_row(self._accumulate()["confusion"])

    def _group_confusion(self) -> np.ndarray:
        """(5, 5) confusion over the coarse planning groups, collapsed from the fine matrix."""
        confusion = self._accumulate()["confusion"]
        group_confusion = np.zeros((NUM_GROUPS, NUM_GROUPS), dtype=np.int64)
        np.add.at(group_confusion, (CLASS_TO_GROUP[:, None], CLASS_TO_GROUP[None, :]), confusion)
        return group_confusion

    @staticmethod
    def _iou_per_row(confusion: np.ndarray) -> np.ndarray:
        """Per-label IoU of a square confusion matrix; NaN where the label never occurs."""
        true_positive = np.diag(confusion)
        union = confusion.sum(axis=0) + confusion.sum(axis=1) - true_positive
        return np.where(union > 0, true_positive / np.maximum(union, 1), np.nan)

    def _binary_group_IOU(self, group_classes: np.ndarray) -> float:
        """
        Binary IoU of a class subset vs everything else, from the fine confusion
        matrix: any within-subset confusion (e.g. car predicted as truck) counts
        as correct.
        """
        confusion = self._accumulate()["confusion"]
        on = group_classes
        true_positive = confusion[np.ix_(on, on)].sum()
        false_negative = confusion[np.ix_(on, ~on)].sum()
        false_positive = confusion[np.ix_(~on, on)].sum()
        union = true_positive + false_negative + false_positive
        return float(true_positive / union) if union else float("nan")

    # ------------------------------------------------------------------ metrics

    def get_IOU(self, cityscapes_class: int | str) -> float:
        """IoU for one Cityscapes class, by train id (13) or name ("car")."""
        if isinstance(cityscapes_class, str):
            cityscapes_class = CLASS_NAMES.index(cityscapes_class)
        return float(self._per_class_IOU()[cityscapes_class])

    def calculate_mean_IOU(self) -> float:
        """Mean IoU over the 19 Cityscapes classes (absent classes excluded)."""
        return float(np.nanmean(self._per_class_IOU()))

    def calculate_planning_group_mIOU(self) -> float:
        """
        Primary planning-oriented summary: mean IoU over the coarse planning
        groups (drivable, soft-drivable, static obstacle, dynamic object) on the
        group-collapsed confusion. The sky/ignore group's own IoU is discarded,
        but its pixels still count against the other groups' unions.
        """
        per_group = self._iou_per_row(self._group_confusion())
        return float(np.nanmean(np.delete(per_group, SKY_GROUP)))

    def calculate_drivable_IOU(self) -> float:
        """
        Binary IoU of the drivable planning group (currently just "road") vs
        everything else.
        """
        return self._binary_group_IOU(DRIVABLE_CLASSES)

    def calculate_drivable_boundary_IOU(self) -> float:
        """
        Boundary IoU [https://arxiv.org/pdf/2103.16562] of the drivable mask:
        plain IoU restricted to a thin band around each mask's contour, so it
        scores edge quality that region IoU washes out.
        """
        stats = self._accumulate()
        if not stats["boundary_union"]:
            return float("nan")
        return stats["boundary_intersection"] / stats["boundary_union"]

    def calculate_traffic_participant_IOU(self) -> float:
        """
        Binary IoU of all traffic participants collapsed into one mask (person,
        rider, car, truck, bus, train, motorcycle, bicycle). Within-group
        confusion (car predicted as truck) counts as correct: the question is
        whether the representation preserved the location and extent of traffic
        participants, not their identity. Report alongside get_IOU("car"), where
        identity does matter and KITTI has enough pixels to be meaningful.
        """
        return self._binary_group_IOU(TRAFFIC_PARTICIPANT_CLASSES)

    # ------------------------------------------------------------------ qualitative

    def visualise_semantic_segmentation(self, viz_every: int = 100) -> None:
        """Every `viz_every` frames: RGB, OneFormer GT, fine and coarse predictions in one PNG."""
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        for item in range(0, len(self.test_set), viz_every):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained semantic decoder")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--viz-every", type=int, default=100,
                        help="save a figure every N test frames (0 disables)")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model = SemanticDecoder().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"loaded {args.checkpoint} (iter {checkpoint['iteration']}, "
          f"val loss {checkpoint['validation_metrics']['loss']:.4f})")

    dataset = KITTISemanticDataset(TEST_SEQUENCES)

    evaluator = SemanticsEvaluator(dataset, ModelPredictions(model, dataset))
    print("\nper-class IoU:")
    for class_name in CLASS_NAMES:
        iou = evaluator.get_IOU(class_name)
        print(f"  {class_name:<14} " + ("absent" if math.isnan(iou) else f"{iou:.4f}"))
    print(f"\nmean IoU (19-class)           {evaluator.calculate_mean_IOU():.4f}")
    print(f"planning-group mIoU (no sky)  {evaluator.calculate_planning_group_mIOU():.4f}")
    print(f"drivable IoU                  {evaluator.calculate_drivable_IOU():.4f}")
    print(f"drivable boundary IoU         {evaluator.calculate_drivable_boundary_IOU():.4f}")
    print(f"traffic-participant IoU       {evaluator.calculate_traffic_participant_IOU():.4f}")
    print(f"car IoU                       {evaluator.get_IOU('car'):.4f}")

    if args.viz_every:
        evaluator.visualise_semantic_segmentation(args.viz_every)
        print(f"figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
