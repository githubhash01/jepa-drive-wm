"""
Focused evaluation of the action-conditioned V-JEPA 2.1 latent predictors.

The evaluator is deliberately shaped around the final thesis tables. For each
checkpoint it reports three things:

1. Latent prediction error
   - autoregressive rollout L1
   - copy-last persistence L1

2. Decoded task performance
   - non-sky depth AbsRel from the predicted latent
   - planning-group semantic mIoU from the predicted latent

3. Action sensitivity
   - real-action rollout L1
   - zero-action rollout L1
   - zero-action minus real-action L1
   - direct L1 difference between real- and zero-action predictions

All latent quantities are averaged over every latent channel, spatial token,
and evaluated test window, while retaining one value per rollout horizon.
Decoded metrics use the same pixel-pooled accumulators as the standalone depth
and semantic decoder evaluations.

Default outputs in OUTPUTS_DIR/evals_wm:

    metrics.json
        Combined machine-readable results for every evaluated checkpoint.

    summary.md
        The exact thesis-facing tables described above.

    <tag>/metrics.json
        Machine-readable results for one checkpoint.

    depth_rollout.png
        FoundationStereo pseudo-depth versus depth decoded from predicted
        latents over four horizons.

    semantics_rollout.png
        OneFormer planning groups versus planning groups decoded from predicted
        latents over four horizons.

The qualitative figures use one deterministic test window shared by every
evaluated timestep model where possible. The default is the temporal midpoint
of the common fixed-anchor grid; --example-sequence and --example-anchor can
override this choice.

Compatibility note
------------------
projected_predictor_evaluator.py imports several small helpers and style
constants from this module. Those public names are retained here:
FIGURE_ANCHOR_GRID, MAX_FIGURE_COLUMNS, SOURCE_STYLE, VARIANT_STYLE,
_anchor_items, _decode, _future_frame, _save_grid, and frame_stride_for.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from jepa_drive_wm.data.data_interface_rollout import KITTIRolloutDataset
from jepa_drive_wm.data.splits import SPLIT_V1
from jepa_drive_wm.evals.common import (
    DEPTH_CHECKPOINT,
    DEVICE,
    SEMANTICS_CHECKPOINT,
    SERIES,
    WM_CHECKPOINT_DIR,
    describe_checkpoint,
    fmt,
    load_depth_decoder,
    load_semantic_decoder,
    load_world_model,
    markdown_table,
    write_metrics_json,
)
from jepa_drive_wm.evals.depth_evaluator import (
    DepthMetricAccumulator,
    SKY_IGNORE_LUT,
)
from jepa_drive_wm.evals.semantics_evaluator import SemanticMetricAccumulator
from jepa_drive_wm.models.dense_decoders.depth_decoder import DepthDecoder
from jepa_drive_wm.models.dense_decoders.semantic_decoder import SemanticDecoder
from jepa_drive_wm.models.predictors.ac_style.ac_predictor import VJEPA21WorldModel
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.train.train_ac_predictor import (
    MAX_SPEED_MPS,
    action_scale,
    to_grid,
)
from jepa_drive_wm.train.train_depth import (
    MAX_DEPTH,
    MIN_DEPTH,
    predict as predict_depth,
)
from jepa_drive_wm.train.train_semantics import (
    NUM_CLASSES,
    predict as predict_semantics,
)
from jepa_drive_wm.training_utils import autocast
from jepa_drive_wm.viz.visualiser import (
    CLASS_NAMES,
    CLASS_TO_GROUP,
    GROUP_NAMES,
    group_colors,
)


FIGURES_DIR = OUTPUTS_DIR / "evals_wm"
# The trainer writes to checkpoints_wm, but on this machine the trained
# action-conditioned models live in checkpoints_ac.
AC_CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints_ac"
TEST_SEQUENCES = list(SPLIT_V1.test_sequences)
FRAME_PERIOD = KITTIRolloutDataset.FRAME_PERIOD

# Retained for projected_predictor_evaluator.py compatibility.
MAX_FIGURE_COLUMNS = 5
FIGURE_ANCHOR_GRID = 50

VARIANTS = {
    "rollout": "autoregressive rollout",
    "teacher_forced": "one-step (teacher forced)",
    "copy_last": "copy last context frame",
    "copy_previous": "copy previous true frame",
    "rollout_zero_action": "rollout, zero action",
}
SOURCES = {
    "true": "true latent",
    "copy": "copy-last-frame latent",
    "predicted": "predicted latent",
}
VARIANT_STYLE = {
    "rollout": dict(linestyle="-", linewidth=2.0, marker="o", markersize=4),
    "teacher_forced": dict(linestyle="--", linewidth=1.6, marker="s", markersize=3.5),
    "copy_last": dict(linestyle=":", linewidth=1.8, marker="^", markersize=3.5),
    "copy_previous": dict(linestyle=":", linewidth=1.2, marker="v", markersize=3.5),
    "rollout_zero_action": dict(linestyle="-.", linewidth=1.2, marker="x", markersize=3.5),
}
SOURCE_STYLE = {
    "predicted": dict(linestyle="-", linewidth=2.0, marker="o", markersize=4),
    "copy": dict(linestyle=":", linewidth=1.8, marker="^", markersize=3.5),
    "true": dict(linestyle="--", linewidth=1.2, marker="s", markersize=3),
}

# LaTeX-like typography without requiring an installed TeX distribution.
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
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
    }
)


# -----------------------------------------------------------------------------
# Checkpoints and shared indexing helpers
# -----------------------------------------------------------------------------


def default_checkpoints() -> list[Path]:
    """Return every standard timestep-tagged action-conditioned checkpoint.

    The trainer writes to WM_CHECKPOINT_DIR (checkpoints_wm), but the trained
    models may live in checkpoints_ac instead; the first directory that
    contains world_model_dt*.pt wins.
    """
    for directory in (WM_CHECKPOINT_DIR, AC_CHECKPOINT_DIR):
        found = sorted(directory.glob("world_model_dt*.pt"))
        if found:
            return found
    return []


def model_tag(checkpoint: dict, path: Path) -> str:
    """Create a stable output tag without allowing same-dt runs to collide."""
    base = f"dt{float(checkpoint['step_seconds']):g}s"
    return base if path.stem == f"world_model_{base}" else f"{base}_{path.stem}"


def frame_stride_for(step_seconds: float) -> int:
    """Convert a physical model step into a whole KITTI frame stride."""
    stride = float(step_seconds) / FRAME_PERIOD
    if not math.isclose(stride, round(stride), abs_tol=1e-6):
        raise ValueError(
            f"step {step_seconds}s is not a whole number of "
            f"{FRAME_PERIOD}s KITTI frames"
        )
    return int(round(stride))


def _future_frame(dataset: KITTIRolloutDataset, start_index: int, k: int) -> int:
    """Raw KITTI frame targeted by zero-based rollout step ``k``."""
    return start_index + (dataset.context_length + k) * dataset.frame_stride


def _anchor_items(dataset: KITTIRolloutDataset, every: int) -> list[int]:
    """
    Return windows whose final context frame lies on a fixed raw-frame grid.

    This makes different timestep models use the same anchor frames and, at
    common physical horizons, the same target frames.
    """
    if every <= 0:
        raise ValueError("every must be positive")

    lookup = {key: item for item, key in enumerate(dataset.index)}
    lead = (dataset.context_length - 1) * dataset.frame_stride
    return [
        lookup[(sequence_nr, anchor - lead)]
        for sequence_nr, sequence in dataset.sequences.items()
        for anchor in range(0, len(sequence), every)
        if (sequence_nr, anchor - lead) in lookup
    ]


@torch.no_grad()
def _decode(
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
    latent_chw: torch.Tensor,
    depth_shape: tuple[int, int],
    semantics_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one V-JEPA latent grid into metric depth and semantic class ids."""
    depth = predict_depth(depth_decoder, latent_chw, depth_shape).cpu().numpy()
    semantics = predict_semantics(
        semantic_decoder, latent_chw, semantics_shape
    ).argmax(dim=1)[0]
    return depth, semantics.to(torch.uint8).cpu().numpy()


def _save_grid(rows, column_titles: list[str], title: str, path: Path) -> None:
    """
    Legacy grid helper retained for projected_predictor_evaluator.py.

    ``rows`` is a sequence of ``(label, images, imshow_kwargs)`` triples.
    """
    n_columns = len(column_titles)
    fig, axes = plt.subplots(
        len(rows),
        n_columns,
        figsize=(6 * n_columns, 2.1 * len(rows)),
        squeeze=False,
    )
    for row, (label, images, kwargs) in enumerate(rows):
        for column in range(n_columns):
            axis = axes[row, column]
            axis.imshow(images[column], **kwargs)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(column_titles[column])
            if column == 0:
                axis.set_ylabel(
                    label,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@dataclass
class ModelSpec:
    path: Path
    tag: str
    step_seconds: float
    frame_stride: int
    dataset: KITTIRolloutDataset
    checkpoint_metadata: dict[str, Any]
    example_item: int | None = None


def prepare_specs(
    checkpoint_paths: list[Path],
    *,
    context_length: int,
    rollout_steps: int,
) -> list[ModelSpec]:
    """Read lightweight checkpoint metadata and build each evaluation dataset."""
    specs: list[ModelSpec] = []
    seen_tags: set[str] = set()

    for path in checkpoint_paths:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if "step_seconds" not in checkpoint:
            raise KeyError(f"{path} does not record step_seconds")

        tag = model_tag(checkpoint, path)
        if tag in seen_tags:
            raise ValueError(f"duplicate model tag {tag!r}; outputs would collide")
        seen_tags.add(tag)

        step_seconds = float(checkpoint["step_seconds"])
        frame_stride = frame_stride_for(step_seconds)
        recorded_stride = checkpoint.get("frame_stride")
        if recorded_stride is not None and int(recorded_stride) != frame_stride:
            raise ValueError(
                f"{path}: recorded frame_stride={recorded_stride}, "
                f"but step_seconds implies {frame_stride}"
            )

        dataset = KITTIRolloutDataset(
            TEST_SEQUENCES,
            context_length=context_length,
            future_length=rollout_steps,
            frame_stride=frame_stride,
        )
        if not math.isclose(dataset.step_seconds, step_seconds, abs_tol=1e-8):
            raise RuntimeError(
                f"{path}: dataset reports {dataset.step_seconds}s but "
                f"checkpoint reports {step_seconds}s"
            )

        metadata = {
            "iteration": checkpoint.get("iteration"),
            "val_ar": checkpoint.get("val_ar"),
            "max_speed_mps": checkpoint.get("max_speed_mps"),
        }
        specs.append(
            ModelSpec(
                path=path,
                tag=tag,
                step_seconds=step_seconds,
                frame_stride=frame_stride,
                dataset=dataset,
                checkpoint_metadata=metadata,
            )
        )
        del checkpoint

    return specs


def _anchor_lookup(dataset: KITTIRolloutDataset) -> dict[tuple[int, int], int]:
    """Map ``(sequence, last_context_frame)`` to dataset item."""
    lead = (dataset.context_length - 1) * dataset.frame_stride
    return {
        (int(sequence), int(start + lead)): item
        for item, (sequence, start) in enumerate(dataset.index)
    }


def assign_shared_example(
    specs: list[ModelSpec],
    *,
    requested_sequence: int | None,
    requested_anchor: int | None,
) -> tuple[int, int]:
    """
    Choose one deterministic ``(sequence, anchor frame)`` available to all models.

    Unless explicitly overridden, the midpoint of the common anchors lying on
    FIGURE_ANCHOR_GRID is selected.
    """
    if not specs:
        raise ValueError("no model specifications supplied")

    lookups = [_anchor_lookup(spec.dataset) for spec in specs]
    common = set(lookups[0])
    for lookup in lookups[1:]:
        common.intersection_update(lookup)

    if requested_sequence is not None:
        common = {key for key in common if key[0] == requested_sequence}
    if requested_anchor is not None:
        common = {key for key in common if key[1] == requested_anchor}

    if not common:
        details = []
        if requested_sequence is not None:
            details.append(f"sequence={requested_sequence}")
        if requested_anchor is not None:
            details.append(f"anchor={requested_anchor}")
        suffix = f" for {', '.join(details)}" if details else ""
        raise RuntimeError(f"no qualitative anchor is shared by all models{suffix}")

    grid_candidates = sorted(
        key for key in common if key[1] % FIGURE_ANCHOR_GRID == 0
    )
    candidates = grid_candidates or sorted(common)
    chosen = candidates[len(candidates) // 2]

    for spec, lookup in zip(specs, lookups):
        spec.example_item = lookup[chosen]
    return chosen


# -----------------------------------------------------------------------------
# Latent evaluation
# -----------------------------------------------------------------------------


class FocusedLatentAccumulator:
    """Dataset-level per-horizon latent metrics for the final thesis tables."""

    KEYS = (
        "autoregressive_l1",
        "copy_last_l1",
        "zero_action_l1",
        "action_sensitivity_l1",
    )

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.sums = {
            key: np.zeros(steps, dtype=np.float64)
            for key in self.KEYS
        }
        self.windows = 0

    def update(
        self,
        *,
        real: torch.Tensor,
        zero: torch.Tensor,
        copy: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Add one batch.

        Tensors have shape ``[B,K,H,W,C]``. Every quantity is first averaged
        over H, W and C, leaving one value per window and horizon.
        """
        if not (real.shape == zero.shape == copy.shape == target.shape):
            raise ValueError(
                "latent tensors must have identical shapes: "
                f"real={tuple(real.shape)}, zero={tuple(zero.shape)}, "
                f"copy={tuple(copy.shape)}, target={tuple(target.shape)}"
            )

        values = {
            "autoregressive_l1": (real.float() - target.float())
            .abs()
            .mean(dim=(2, 3, 4)),
            "copy_last_l1": (copy.float() - target.float())
            .abs()
            .mean(dim=(2, 3, 4)),
            "zero_action_l1": (zero.float() - target.float())
            .abs()
            .mean(dim=(2, 3, 4)),
            "action_sensitivity_l1": (real.float() - zero.float())
            .abs()
            .mean(dim=(2, 3, 4)),
        }

        batch_size = int(real.shape[0])
        for key, tensor in values.items():
            self.sums[key] += tensor.detach().cpu().double().sum(dim=0).numpy()
        self.windows += batch_size

    def summary(self) -> dict[str, Any]:
        divisor = max(self.windows, 1)
        means = {
            key: (value / divisor).tolist()
            for key, value in self.sums.items()
        }
        means["zero_minus_real_l1"] = (
            np.asarray(means["zero_action_l1"])
            - np.asarray(means["autoregressive_l1"])
        ).tolist()
        return {"windows": self.windows, **means}


@torch.no_grad()
def evaluate_latent(
    model: VJEPA21WorldModel,
    loader: DataLoader,
    *,
    act_scale: torch.Tensor,
    steps: int,
) -> dict[str, Any]:
    """Evaluate real-action, zero-action and copy-last predictions."""
    model.eval()
    accumulator = FocusedLatentAccumulator(steps)
    height, width = model.grid_height, model.grid_width

    for batch in loader:
        context = to_grid(
            batch["context_latents"].to(DEVICE, non_blocking=True),
            height,
            width,
        )
        target = to_grid(
            batch["future_latents"].to(DEVICE, non_blocking=True),
            height,
            width,
        )
        actions = batch["future_ego_motions"].to(DEVICE) / act_scale

        with autocast(DEVICE):
            real = model.rollout(context, actions)
            zero = model.rollout(context, torch.zeros_like(actions))

        copy = context[:, -1:].expand_as(target)
        accumulator.update(
            real=real[:, :steps],
            zero=zero[:, :steps],
            copy=copy[:, :steps],
            target=target[:, :steps],
        )

    return accumulator.summary()


# -----------------------------------------------------------------------------
# Decoded evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_decoded(
    model: VJEPA21WorldModel,
    dataset: KITTIRolloutDataset,
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
    *,
    act_scale: torch.Tensor,
    steps: int,
    every: int,
    max_windows: int | None,
) -> dict[str, Any]:
    """
    Decode only the autoregressively predicted latents.

    Each horizon has an independent depth and semantic accumulator, so the
    reported metrics are pixel-pooled over all selected target frames at that
    horizon.
    """
    model.eval()
    height, width = model.grid_height, model.grid_width
    depth_accumulators = [DepthMetricAccumulator() for _ in range(steps)]
    semantic_accumulators = [SemanticMetricAccumulator() for _ in range(steps)]

    items = _anchor_items(dataset, every)
    if max_windows is not None:
        items = items[:max_windows]

    for number, item in enumerate(items, start=1):
        sample = dataset[item]
        sequence_nr = int(sample["sequence_nr"].item())
        sequence = dataset.sequences[sequence_nr]
        start_index = int(sample["start_index"].item())

        context = to_grid(
            sample["context_latents"][None].to(DEVICE),
            height,
            width,
        )
        actions = sample["future_ego_motions"][None].to(DEVICE) / act_scale

        with autocast(DEVICE):
            predictions = model.rollout(context, actions)[0].float()

        for step in range(steps):
            frame = _future_frame(dataset, start_index, step)
            target_depth = sequence.get_depth(frame)
            target_semantics = sequence.get_semantics(frame)
            latent = predictions[step].permute(2, 0, 1)[None]

            decoded_depth, decoded_semantics = _decode(
                depth_decoder,
                semantic_decoder,
                latent,
                target_depth.shape,
                target_semantics.shape,
            )
            depth_accumulators[step].update(
                target_depth,
                decoded_depth,
                target_semantics,
            )
            semantic_accumulators[step].update(
                target_semantics,
                decoded_semantics,
            )

        if number % 25 == 0 or number == len(items):
            print(f"  decoded {number}/{len(items)} windows", flush=True)

    depth_absrel = [
        float(accumulator.summary()["non-sky"]["absrel"])
        for accumulator in depth_accumulators
    ]
    planning_miou = [
        float(accumulator.summary()["planning_group_miou"])
        for accumulator in semantic_accumulators
    ]

    return {
        "windows": len(items),
        "every": every,
        "aggregation": (
            "one pixel-pooled accumulator per horizon over fixed-grid anchor windows"
        ),
        "depth_non_sky_absrel": depth_absrel,
        "semantics_planning_group_miou": planning_miou,
    }


# -----------------------------------------------------------------------------
# Qualitative evaluation
# -----------------------------------------------------------------------------


def _normalise_colour(pixel: np.ndarray) -> tuple[float, float, float]:
    colour = np.asarray(pixel, dtype=np.float64).reshape(-1)[:3]
    if colour.max(initial=0.0) > 1.0:
        colour = colour / 255.0
    return tuple(float(value) for value in colour)


def _planning_colour_image(labels: np.ndarray) -> np.ndarray:
    """Render fine Cityscapes ids using the project's planning-group colours."""
    labels = labels.astype(np.int64, copy=False)
    labelled = (labels >= 0) & (labels < NUM_CLASSES)
    sky_class = CLASS_NAMES.index("sky")
    safe = np.where(labelled, labels, sky_class).astype(np.uint8)
    return np.asarray(group_colors(safe))


def _planning_legend_handles() -> list[Patch]:
    handles: list[Patch] = []
    for group_id, name in enumerate(GROUP_NAMES):
        members = np.flatnonzero(CLASS_TO_GROUP == group_id)
        if members.size == 0:
            continue
        sample = group_colors(
            np.array([[members[0]]], dtype=np.uint8)
        )[0, 0]
        handles.append(
            Patch(
                facecolor=_normalise_colour(sample),
                edgecolor="none",
                label=name,
            )
        )
    return handles


def _non_sky_display_mask(
    depth: np.ndarray,
    semantics: np.ndarray,
) -> np.ndarray:
    """Use the same depth range and semantic sky exclusion as depth evaluation."""
    in_range = (
        np.isfinite(depth)
        & (depth > MIN_DEPTH)
        & (depth < MAX_DEPTH)
    )
    return in_range & ~SKY_IGNORE_LUT[semantics]


@torch.no_grad()
def collect_qualitative_example(
    model: VJEPA21WorldModel,
    spec: ModelSpec,
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
    *,
    act_scale: torch.Tensor,
    steps: int,
) -> dict[str, Any]:
    """Collect one predicted rollout and its paired pseudo-labels on CPU."""
    if spec.example_item is None:
        raise RuntimeError("qualitative example item has not been assigned")

    dataset = spec.dataset
    sample = dataset[spec.example_item]
    sequence_nr = int(sample["sequence_nr"].item())
    start_index = int(sample["start_index"].item())
    sequence = dataset.sequences[sequence_nr]

    height, width = model.grid_height, model.grid_width
    context = to_grid(
        sample["context_latents"][None].to(DEVICE),
        height,
        width,
    )
    actions = sample["future_ego_motions"][None].to(DEVICE) / act_scale

    with autocast(DEVICE):
        predictions = model.rollout(context, actions)[0].float()

    target_depths: list[np.ndarray] = []
    predicted_depths: list[np.ndarray] = []
    depth_masks: list[np.ndarray] = []
    target_semantics: list[np.ndarray] = []
    predicted_semantics: list[np.ndarray] = []
    future_frames: list[int] = []

    for step in range(steps):
        frame = _future_frame(dataset, start_index, step)
        target_depth = sequence.get_depth(frame)
        target_semantic = sequence.get_semantics(frame)
        latent = predictions[step].permute(2, 0, 1)[None]

        decoded_depth, decoded_semantic = _decode(
            depth_decoder,
            semantic_decoder,
            latent,
            target_depth.shape,
            target_semantic.shape,
        )

        target_depths.append(np.asarray(target_depth))
        predicted_depths.append(np.asarray(decoded_depth))
        depth_masks.append(_non_sky_display_mask(target_depth, target_semantic))
        target_semantics.append(np.asarray(target_semantic))
        predicted_semantics.append(np.asarray(decoded_semantic))
        future_frames.append(int(frame))

    anchor_frame = _future_frame(dataset, start_index, -1)
    return {
        "tag": spec.tag,
        "step_seconds": spec.step_seconds,
        "horizons": [
            (step + 1) * spec.step_seconds
            for step in range(steps)
        ],
        "sequence": sequence_nr,
        "anchor_frame": int(anchor_frame),
        "future_frames": future_frames,
        "target_depths": target_depths,
        "predicted_depths": predicted_depths,
        "depth_masks": depth_masks,
        "target_semantics": target_semantics,
        "predicted_semantics": predicted_semantics,
    }


def plot_depth_rollouts(
    examples: list[dict[str, Any]],
    path: Path,
) -> Path:
    """Two rows per model: FoundationStereo and predicted-latent depth."""
    if not examples:
        raise ValueError("no qualitative examples supplied")

    steps = len(examples[0]["horizons"])
    rows = 2 * len(examples)
    fig, axes = plt.subplots(
        rows,
        steps,
        figsize=(3.15 * steps + 0.6, 1.78 * rows + 0.65),
        squeeze=False,
    )

    colour_map = plt.get_cmap("plasma").copy()
    colour_map.set_bad("white")
    normalisation = LogNorm(vmin=MIN_DEPTH, vmax=MAX_DEPTH)
    last_image = None

    for model_index, example in enumerate(examples):
        target_row = 2 * model_index
        prediction_row = target_row + 1

        for step in range(steps):
            mask = example["depth_masks"][step]
            target = np.ma.masked_where(
                ~mask,
                example["target_depths"][step],
            )
            prediction = np.ma.masked_where(
                ~mask,
                example["predicted_depths"][step],
            )

            axes[target_row, step].imshow(
                target,
                cmap=colour_map,
                norm=normalisation,
            )
            last_image = axes[prediction_row, step].imshow(
                prediction,
                cmap=colour_map,
                norm=normalisation,
            )

            axes[target_row, step].set_title(
                f"$t+{example['horizons'][step]:.1f}\\,\\mathrm{{s}}$",
                pad=6,
            )

            for row in (target_row, prediction_row):
                axis = axes[row, step]
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)

        model_label = (
            f"{example['step_seconds']:g} s-step model\n"
            f"seq. {example['sequence']:02d}, anchor {example['anchor_frame']}"
        )
        axes[target_row, 0].set_ylabel(
            model_label + "\nFoundationStereo",
            rotation=0,
            ha="right",
            va="center",
            labelpad=16,
            fontsize=8.5,
        )
        axes[prediction_row, 0].set_ylabel(
            "Predicted-latent\ndecode",
            rotation=0,
            ha="right",
            va="center",
            labelpad=16,
            fontsize=8.5,
        )

    fig.suptitle(
        "Depth decoded from autoregressively predicted V-JEPA latents",
        x=0.08,
        ha="left",
        fontsize=11,
    )
    fig.subplots_adjust(
        left=0.17,
        right=0.91,
        top=0.92,
        bottom=0.06,
        wspace=0.025,
        hspace=0.08,
    )

    # The separators between model blocks are drawn only now, after
    # subplots_adjust has fixed the layout (drawing them earlier would pin
    # them to stale axis positions): measured from the rendered artists,
    # each line sits midway between the lower model's column titles and the
    # upper model's letterboxed image panels.
    if len(examples) > 1:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_figure = fig.transFigure.inverted()
        for model_index in range(1, len(examples)):
            target_row = 2 * model_index
            title_top = max(
                to_figure.transform(
                    (0, axes[target_row, step].title.get_window_extent(renderer).y1)
                )[1]
                for step in range(steps)
            )
            image_bottom = min(
                to_figure.transform(
                    (0, axes[target_row - 1, step].images[0].get_window_extent(renderer).y0)
                )[1]
                for step in range(steps)
            )
            separator_y = (title_top + image_bottom) / 2
            fig.add_artist(
                plt.Line2D(
                    [0.08, 0.93],
                    [separator_y, separator_y],
                    transform=fig.transFigure,
                    color="#d0d0d0",
                    linewidth=0.8,
                )
            )

    if last_image is not None:
        colour_axis = fig.add_axes([0.925, 0.085, 0.016, 0.78])
        colour_bar = fig.colorbar(last_image, cax=colour_axis)
        colour_bar.set_label("Depth (m, logarithmic scale)", fontsize=8.5)
        colour_bar.set_ticks([1, 2, 5, 10, 20, 40, 80])
        colour_bar.set_ticklabels(["1", "2", "5", "10", "20", "40", "80"])

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_semantic_rollouts(
    examples: list[dict[str, Any]],
    path: Path,
) -> Path:
    """Two rows per model: OneFormer and predicted-latent planning groups."""
    if not examples:
        raise ValueError("no qualitative examples supplied")

    steps = len(examples[0]["horizons"])
    rows = 2 * len(examples)
    fig, axes = plt.subplots(
        rows,
        steps,
        figsize=(3.15 * steps + 0.6, 1.78 * rows + 0.95),
        squeeze=False,
    )

    for model_index, example in enumerate(examples):
        target_row = 2 * model_index
        prediction_row = target_row + 1

        for step in range(steps):
            axes[target_row, step].imshow(
                _planning_colour_image(example["target_semantics"][step])
            )
            axes[prediction_row, step].imshow(
                _planning_colour_image(example["predicted_semantics"][step])
            )
            axes[target_row, step].set_title(
                f"$t+{example['horizons'][step]:.1f}\\,\\mathrm{{s}}$",
                pad=6,
            )

            for row in (target_row, prediction_row):
                axis = axes[row, step]
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)

        model_label = (
            f"{example['step_seconds']:g} s-step model\n"
            f"seq. {example['sequence']:02d}, anchor {example['anchor_frame']}"
        )
        axes[target_row, 0].set_ylabel(
            model_label + "\nOneFormer",
            rotation=0,
            ha="right",
            va="center",
            labelpad=16,
            fontsize=8.5,
        )
        axes[prediction_row, 0].set_ylabel(
            "Predicted-latent\ndecode",
            rotation=0,
            ha="right",
            va="center",
            labelpad=16,
            fontsize=8.5,
        )

    handles = _planning_legend_handles()
    legend_rows = max(1, math.ceil(len(handles) / 5))
    bottom = 0.07 + 0.035 * legend_rows

    fig.suptitle(
        "Planning groups decoded from autoregressively predicted V-JEPA latents",
        x=0.08,
        ha="left",
        fontsize=11,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(5, len(handles)),
        frameon=False,
        columnspacing=1.3,
        handlelength=1.4,
        bbox_to_anchor=(0.55, 0.01),
    )
    fig.subplots_adjust(
        left=0.17,
        right=0.995,
        top=0.92,
        bottom=bottom,
        wspace=0.025,
        hspace=0.08,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def _horizon_label(value: float) -> str:
    return f"t+{value:.1f} s"


def summary_markdown(results: dict[str, dict[str, Any]]) -> str:
    """Create the exact focused thesis-facing Markdown tables."""
    lines = [
        "# Action-conditioned latent predictors -- held-out test set",
        "",
        "test sequences: "
        + ", ".join(f"{sequence:02d}" for sequence in TEST_SEQUENCES),
        "",
        "Latent L1 is the mean absolute difference over every spatial patch, "
        "all 768 feature channels, and all evaluated test windows, reported "
        "separately at each rollout horizon.",
        "",
    ]

    for tag, result in results.items():
        latent = result["latent"]
        horizons = result["horizon_seconds"]
        lines.extend(
            [
                f"## {result['step_seconds']:g} s-step model (`{tag}`)",
                "",
                f"checkpoint: `{result['checkpoint']}`  ",
                f"context: {result['context_length']} frames; "
                f"rollout: {result['horizon_steps']} steps; "
                f"latent windows: {latent['windows']}",
                "",
                "### Latent prediction error",
                "",
                markdown_table(
                    [
                        "Prediction horizon",
                        "Autoregressive latent L1 ↓",
                        "Copy-last latent L1 ↓",
                    ],
                    [
                        [
                            _horizon_label(horizon),
                            latent["autoregressive_l1"][index],
                            latent["copy_last_l1"][index],
                        ]
                        for index, horizon in enumerate(horizons)
                    ],
                ),
                "",
            ]
        )

        decoded = result.get("decoded")
        if decoded is not None:
            lines.extend(
                [
                    "### Decoded performance: depth",
                    "",
                    markdown_table(
                        [
                            "Prediction horizon",
                            "Predicted-latent non-sky AbsRel ↓",
                        ],
                        [
                            [
                                _horizon_label(horizon),
                                decoded["depth_non_sky_absrel"][index],
                            ]
                            for index, horizon in enumerate(horizons)
                        ],
                    ),
                    "",
                    "### Decoded performance: semantic segmentation",
                    "",
                    markdown_table(
                        [
                            "Prediction horizon",
                            "Predicted-latent planning-group mIoU ↑",
                        ],
                        [
                            [
                                _horizon_label(horizon),
                                decoded["semantics_planning_group_miou"][index],
                            ]
                            for index, horizon in enumerate(horizons)
                        ],
                    ),
                    "",
                    f"Decoded metrics use {decoded['windows']} fixed-grid test "
                    f"windows (every {decoded['every']} raw frames) and are "
                    "pooled over pixels separately at each horizon.",
                    "",
                ]
            )

        lines.extend(
            [
                "### Action sensitivity",
                "",
                markdown_table(
                    [
                        "Prediction horizon",
                        "Real-action L1 ↓",
                        "Zero-action L1 ↓",
                        "Zero − real L1 ↑",
                        "Action sensitivity",
                    ],
                    [
                        [
                            _horizon_label(horizon),
                            latent["autoregressive_l1"][index],
                            latent["zero_action_l1"][index],
                            latent["zero_minus_real_l1"][index],
                            latent["action_sensitivity_l1"][index],
                        ]
                        for index, horizon in enumerate(horizons)
                    ],
                ),
                "",
                "`Zero − real L1` is positive when supplying the recorded ego "
                "motion improves target prediction. `Action sensitivity` is the "
                "direct mean absolute difference between the real-action and "
                "zero-action predicted latents.",
                "",
            ]
        )

        example = result.get("qualitative_example")
        if example:
            lines.extend(
                [
                    "### Qualitative example",
                    "",
                    f"Sequence {example['sequence']:02d}, final context frame "
                    f"{example['anchor_frame']}; future frames: "
                    + ", ".join(str(frame) for frame in example["future_frames"])
                    + ".",
                    "",
                ]
            )

    return "\n".join(lines)


def print_overview(tag: str, result: dict[str, Any]) -> None:
    """Concise console view of the same core quantities."""
    latent = result["latent"]
    decoded = result.get("decoded")
    print(f"\n[{tag}] {result['step_seconds']:g} s-step model")
    print(
        f"{'horizon':<10} {'AR L1':>9} {'copy L1':>9} "
        f"{'zero L1':>9} {'zero-real':>10} {'sensitivity':>12}"
    )
    for index, horizon in enumerate(result["horizon_seconds"]):
        print(
            f"{_horizon_label(horizon):<10} "
            f"{fmt(latent['autoregressive_l1'][index]):>9} "
            f"{fmt(latent['copy_last_l1'][index]):>9} "
            f"{fmt(latent['zero_action_l1'][index]):>9} "
            f"{fmt(latent['zero_minus_real_l1'][index]):>10} "
            f"{fmt(latent['action_sensitivity_l1'][index]):>12}"
        )

    if decoded:
        print("\nDecoded predicted-latent metrics")
        print(f"{'horizon':<10} {'AbsRel':>10} {'planning mIoU':>16}")
        for index, horizon in enumerate(result["horizon_seconds"]):
            print(
                f"{_horizon_label(horizon):<10} "
                f"{fmt(decoded['depth_non_sky_absrel'][index]):>10} "
                f"{fmt(decoded['semantics_planning_group_miou'][index]):>16}"
            )


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------


def evaluate_spec(
    spec: ModelSpec,
    args: argparse.Namespace,
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load one model, run every requested evaluation, then release it."""
    model, checkpoint = load_world_model(spec.path)
    checkpoint_line = describe_checkpoint(spec.path, checkpoint)

    recorded_max_speed = checkpoint.get("max_speed_mps")
    if recorded_max_speed is not None and not math.isclose(
        float(recorded_max_speed),
        MAX_SPEED_MPS,
    ):
        raise ValueError(
            f"{spec.path}: checkpoint MAX_SPEED_MPS={recorded_max_speed}, "
            f"current code uses {MAX_SPEED_MPS}"
        )

    print(
        f"\nloaded {checkpoint_line}\n"
        f"[{spec.tag}] {len(spec.dataset)} test windows; "
        f"{args.rollout_steps} steps of {spec.step_seconds:g}s"
    )

    latent_items: Any = range(len(spec.dataset))
    if args.max_windows is not None:
        latent_items = range(min(args.max_windows, len(spec.dataset)))
    latent_loader = DataLoader(
        Subset(spec.dataset, list(latent_items)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    act_scale = action_scale(spec.step_seconds).to(DEVICE)
    latent = evaluate_latent(
        model,
        latent_loader,
        act_scale=act_scale,
        steps=args.rollout_steps,
    )

    result: dict[str, Any] = {
        "checkpoint": checkpoint_line,
        "checkpoint_path": str(spec.path),
        "iteration": checkpoint.get("iteration"),
        "val_ar": checkpoint.get("val_ar"),
        "step_seconds": spec.step_seconds,
        "frame_stride": spec.frame_stride,
        "context_length": args.context_length,
        "horizon_steps": args.rollout_steps,
        "horizon_seconds": [
            (step + 1) * spec.step_seconds
            for step in range(args.rollout_steps)
        ],
        "test_sequences": TEST_SEQUENCES,
        "latent": latent,
    }

    if not args.skip_decoded:
        print(
            f"[{spec.tag}] decoding every {args.decoded_every}th anchor window ..."
        )
        result["decoded"] = evaluate_decoded(
            model,
            spec.dataset,
            depth_decoder,
            semantic_decoder,
            act_scale=act_scale,
            steps=args.rollout_steps,
            every=args.decoded_every,
            max_windows=args.max_windows,
        )

    qualitative = None
    if not args.skip_figures:
        qualitative = collect_qualitative_example(
            model,
            spec,
            depth_decoder,
            semantic_decoder,
            act_scale=act_scale,
            steps=args.rollout_steps,
        )
        result["qualitative_example"] = {
            "sequence": qualitative["sequence"],
            "anchor_frame": qualitative["anchor_frame"],
            "future_frames": qualitative["future_frames"],
        }

    print_overview(spec.tag, result)

    tag_dir = args.figures_dir / spec.tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(result, tag_dir / "metrics.json")

    del model
    del checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, qualitative


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused evaluation of action-conditioned V-JEPA predictors"
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "predictor checkpoints; by default evaluates every "
            "checkpoints_wm/world_model_dt*.pt"
        ),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4,
        help="number of cached V-JEPA context frames",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=4,
        help="number of autoregressive future states to evaluate",
    )
    parser.add_argument(
        "--decoded-every",
        type=int,
        default=20,
        help="decode windows whose final context frame lies every N raw frames",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch size for the full latent-space pass",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="cap latent and decoded windows for a smoke test",
    )
    parser.add_argument(
        "--skip-decoded",
        action="store_true",
        help="evaluate latent quantities only",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="do not generate the two qualitative rollout figures",
    )
    parser.add_argument(
        "--example-sequence",
        type=int,
        default=None,
        help="optional held-out sequence for the shared qualitative example",
    )
    parser.add_argument(
        "--example-anchor",
        type=int,
        default=None,
        help="optional raw frame id of the final context frame",
    )
    parser.add_argument(
        "--depth-checkpoint",
        type=Path,
        default=DEPTH_CHECKPOINT,
    )
    parser.add_argument(
        "--semantics-checkpoint",
        type=Path,
        default=SEMANTICS_CHECKPOINT,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help=(
            "skip evaluation: rebuild summary.md and the combined metrics.json "
            "from the per-model <figures-dir>/<tag>/metrics.json files of an "
            "earlier run (tables only; the qualitative figures need a real run)"
        ),
    )
    args = parser.parse_args()

    if args.rollout_steps <= 0:
        raise SystemExit("--rollout-steps must be positive")
    if args.decoded_every <= 0:
        raise SystemExit("--decoded-every must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    if args.replot:
        results = {
            path.parent.name: json.loads(path.read_text())
            for path in sorted(args.figures_dir.glob("*/metrics.json"))
        }
        # Skip metrics.json files written by the previous evaluator's schema.
        results = {
            tag: result
            for tag, result in results.items()
            if "autoregressive_l1" in result.get("latent", {})
        }
        if not results:
            raise SystemExit(
                f"no compatible <tag>/metrics.json under {args.figures_dir} to replot"
            )
        print("replotting from: " + ", ".join(results))
        for tag, result in results.items():
            print_overview(tag, result)
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        write_metrics_json(
            {
                "evaluation": "focused action-conditioned latent predictor",
                "models": results,
            },
            args.figures_dir / "metrics.json",
        )
        (args.figures_dir / "summary.md").write_text(summary_markdown(results))
        print(f"combined metrics:   {args.figures_dir / 'metrics.json'}")
        print(f"thesis tables:      {args.figures_dir / 'summary.md'}")
        return

    checkpoint_paths = args.checkpoints or default_checkpoints()
    if not checkpoint_paths:
        raise SystemExit(
            "no world-model checkpoints found in "
            f"{WM_CHECKPOINT_DIR} or {AC_CHECKPOINT_DIR}"
        )
    missing = [path for path in checkpoint_paths if not path.exists()]
    if missing:
        raise SystemExit(
            "checkpoint files do not exist: "
            + ", ".join(str(path) for path in missing)
        )

    specs = prepare_specs(
        list(checkpoint_paths),
        context_length=args.context_length,
        rollout_steps=args.rollout_steps,
    )

    if not args.skip_figures:
        sequence, anchor = assign_shared_example(
            specs,
            requested_sequence=args.example_sequence,
            requested_anchor=args.example_anchor,
        )
        print(
            f"qualitative example shared across models: "
            f"sequence {sequence:02d}, anchor frame {anchor}"
        )

    depth_decoder, depth_checkpoint = load_depth_decoder(args.depth_checkpoint)
    semantic_decoder, semantic_checkpoint = load_semantic_decoder(
        args.semantics_checkpoint
    )
    print(
        "depth decoder:    "
        + describe_checkpoint(args.depth_checkpoint, depth_checkpoint)
    )
    print(
        "semantic decoder: "
        + describe_checkpoint(args.semantics_checkpoint, semantic_checkpoint)
    )

    results: dict[str, dict[str, Any]] = {}
    qualitative_examples: list[dict[str, Any]] = []

    for spec in specs:
        result, qualitative = evaluate_spec(
            spec,
            args,
            depth_decoder,
            semantic_decoder,
        )
        results[spec.tag] = result
        if qualitative is not None:
            qualitative_examples.append(qualitative)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "evaluation": "focused action-conditioned latent predictor",
        "models": results,
    }
    write_metrics_json(combined, args.figures_dir / "metrics.json")
    (args.figures_dir / "summary.md").write_text(
        summary_markdown(results)
    )

    if qualitative_examples:
        depth_path = plot_depth_rollouts(
            qualitative_examples,
            args.figures_dir / "depth_rollout.png",
        )
        semantics_path = plot_semantic_rollouts(
            qualitative_examples,
            args.figures_dir / "semantics_rollout.png",
        )
        print(f"\ndepth figure:       {depth_path}")
        print(f"semantics figure:   {semantics_path}")

    print(f"combined metrics:   {args.figures_dir / 'metrics.json'}")
    print(f"thesis tables:      {args.figures_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
