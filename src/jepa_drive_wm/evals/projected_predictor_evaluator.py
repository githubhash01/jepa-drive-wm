"""
Focused evaluation of the geometry-guided ProjectedPredictor world models.

The evaluator deliberately mirrors the final thesis-facing protocol of
``ac_predictor_evaluator.py``.  For each checkpoint it reports three things:

1. Latent prediction error
   - autoregressive learned rollout L1
   - correction-free deterministic latent-proposal rollout L1

2. Decoded task performance
   - non-sky depth AbsRel and planning-group semantic mIoU from the
     predicted latent, next to the same metrics from the cached target
     latent (the frozen decoders' ceiling on the same frames)

3. Action sensitivity
   - real-action rollout L1
   - zero-action rollout L1
   - zero-action minus real-action L1
   - direct L1 difference between real- and zero-action predictions

The deterministic baseline is rebuilt independently of the learned rollout.
At each horizon, warp-valid patches use the frozen image-mode latent of the
RGB-D projection, while missing patches copy the previous *deterministic*
proposal.  A learned prediction is never fed back into this baseline.

All latent errors are measured against the cached ``vjepa_vitb`` encoding of
the actual future frame.  This matches the action-conditioned evaluator and
the latent source on which the frozen dense decoders were trained.

Default outputs in OUTPUTS_DIR/evals_projected_predictor:

    metrics.json
        Combined machine-readable results for every evaluated checkpoint.

    summary.md
        The focused thesis-facing tables.

    <tag>/metrics.json
        Machine-readable results for one checkpoint.

    depth_rollout_<tag>.png
        FoundationStereo pseudo-depth versus depth decoded from predicted
        latents over the evaluated horizons.

    semantics_rollout_<tag>.png
        OneFormer fine-class semantics versus the predicted-latent decode.

    planning_semantics_rollout_<tag>.png
        The same comparison rendered as the coarse planning groups.

    decoded_metrics_vs_horizon_<tag>.png
        Non-sky AbsRel and planning-group mIoU against horizon for the
        predicted latent next to the cached target latent.

The qualitative figures use one deterministic test window shared by every
projected-predictor timestep model where possible.  ``--example-sequence`` and
``--example-anchor`` can override the default midpoint choice.

Example:

    PYTHONPATH=src python -m jepa_drive_wm.evals.projected_predictor_evaluator
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from jepa_drive_wm.data.data_interface_projected_predictor import (
    KITTIProjectedPredictorDataset,
)
from jepa_drive_wm.data.splits import SPLIT_V1
from jepa_drive_wm.evals.ac_predictor_evaluator import (
    FIGURE_ANCHOR_GRID,
    _anchor_items,
    _decode,
    _future_frame,
    _non_sky_display_mask,
    frame_stride_for,
    plot_decoded_curves,
    plot_depth_rollout,
    plot_semantic_rollout,
)
from jepa_drive_wm.evals.common import (
    DEPTH_CHECKPOINT,
    DEVICE,
    PROJECTED_CHECKPOINT_DIR,
    SEMANTICS_CHECKPOINT,
    describe_checkpoint,
    fmt,
    load_depth_decoder,
    load_semantic_decoder,
    markdown_table,
    write_metrics_json,
)
from jepa_drive_wm.evals.depth_evaluator import DepthMetricAccumulator
from jepa_drive_wm.evals.semantics_evaluator import SemanticMetricAccumulator
from jepa_drive_wm.models.dense_decoders.depth_decoder import DepthDecoder
from jepa_drive_wm.models.dense_decoders.semantic_decoder import SemanticDecoder
from jepa_drive_wm.models.predictors.projected_predictor.projected_predictor import (
    ProjectedPredictor,
    ProjectedPredictorOutput,
)
from jepa_drive_wm.models.predictors.projected_predictor.warp_rgb import (
    DEFAULT_PATCH_COVERAGE_THRESHOLD,
    DEFAULT_RADIUS_PX,
)
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.train.train_projected_predictor import (
    DEFAULT_VJEPA_CHECKPOINT,
    IMAGE_SIZE,
    PATCH_SIZE,
    build_model,
)
from jepa_drive_wm.training_utils import autocast


FIGURES_DIR = OUTPUTS_DIR / "evals_projected_predictor"
TEST_SEQUENCES = list(SPLIT_V1.test_sequences)
CONTEXT_LENGTH = 5
MAX_STEPS = ProjectedPredictor.NUM_FUTURE_STEPS
GRID_HEIGHT = IMAGE_SIZE[0] // PATCH_SIZE
GRID_WIDTH = IMAGE_SIZE[1] // PATCH_SIZE
NUM_TOKENS = GRID_HEIGHT * GRID_WIDTH


# -----------------------------------------------------------------------------
# Checkpoints and indexing
# -----------------------------------------------------------------------------


def default_checkpoints() -> list[Path]:
    """Return every standard projected-predictor timestep checkpoint."""
    return sorted(PROJECTED_CHECKPOINT_DIR.glob("projected_predictor_dt*.pt"))


def model_tag(checkpoint: dict[str, Any], path: Path) -> str:
    """Create a stable output tag without allowing same-dt runs to collide."""
    base = f"dt{float(checkpoint['step_seconds']):g}s"
    return (
        base
        if path.stem == f"projected_predictor_{base}"
        else f"{base}_{path.stem}"
    )


@dataclass
class ModelSpec:
    path: Path
    tag: str
    step_seconds: float
    frame_stride: int
    dataset: KITTIProjectedPredictorDataset
    example_item: int | None = None


def prepare_specs(checkpoint_paths: list[Path]) -> list[ModelSpec]:
    """Read checkpoint metadata and construct each projected test dataset."""
    specs: list[ModelSpec] = []
    seen_tags: set[str] = set()

    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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

        # The architecture always consumes four source-relative future motions,
        # even when --rollout-steps asks us to score only an initial prefix.
        dataset = KITTIProjectedPredictorDataset(
            TEST_SEQUENCES,
            context_length=CONTEXT_LENGTH,
            future_length=MAX_STEPS,
            frame_stride=frame_stride,
            image_size=IMAGE_SIZE,
        )
        if not math.isclose(dataset.step_seconds, step_seconds, abs_tol=1e-8):
            raise RuntimeError(
                f"{path}: dataset reports {dataset.step_seconds}s but "
                f"checkpoint reports {step_seconds}s"
            )

        specs.append(
            ModelSpec(
                path=path,
                tag=tag,
                step_seconds=step_seconds,
                frame_stride=frame_stride,
                dataset=dataset,
            )
        )
        del checkpoint

    return specs


def _anchor_lookup(
    dataset: KITTIProjectedPredictorDataset,
) -> dict[tuple[int, int], int]:
    """Map ``(sequence, final-context frame)`` to a dataset item."""
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
    """Choose one deterministic qualitative anchor shared by every model."""
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
        details: list[str] = []
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
# Model and latent helpers
# -----------------------------------------------------------------------------


def _resolve_vjepa_checkpoint(
    checkpoint: dict[str, Any],
    explicit: Path | None,
) -> Path:
    """Resolve the released V-JEPA checkpoint with a clear failure message."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"V-JEPA checkpoint does not exist: {explicit}")
        return explicit

    recorded_value = checkpoint.get("base_vjepa_checkpoint")
    default_path = Path(DEFAULT_VJEPA_CHECKPOINT)
    candidates: list[Path] = []
    if recorded_value:
        candidates.append(Path(recorded_value))
    if default_path not in candidates:
        candidates.append(default_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "could not find the released V-JEPA checkpoint; tried: "
        + ", ".join(str(path) for path in candidates)
        + ". Pass --vjepa-checkpoint explicitly."
    )


def load_projected_predictor(
    path: Path,
    *,
    vjepa_checkpoint: Path | None,
    radius_px: float,
    patch_coverage_threshold: float,
) -> tuple[ProjectedPredictor, dict[str, Any]]:
    """Rebuild one trained ProjectedPredictor and load its learned weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.pop("optimizer", None)

    resolved_vjepa = _resolve_vjepa_checkpoint(checkpoint, vjepa_checkpoint)
    model = build_model(
        device=DEVICE,
        vjepa_checkpoint=resolved_vjepa,
        radius_px=radius_px,
        patch_coverage_threshold=patch_coverage_threshold,
        finetune_predictor=False,
    )
    model.predictor.load_state_dict(checkpoint.pop("predictor"))
    model.correction_head.load_state_dict(checkpoint.pop("correction_head"))
    model.eval()

    checkpoint["vjepa_checkpoint_used"] = str(resolved_vjepa)
    return model, checkpoint


def _unbatch(batch: dict[str, Any]) -> dict[str, Any]:
    """Remove the size-one DataLoader batch axis used by this per-sample model."""
    sample: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, (torch.Tensor, np.ndarray, list, tuple)):
            sample[key] = value[0]
        else:
            sample[key] = value
    return sample


def _as_int(value: Any) -> int:
    """Convert a Python/numpy/torch scalar to ``int``."""
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


@torch.no_grad()
def run_model(
    model: ProjectedPredictor,
    sample: dict[str, Any],
    *,
    ego_motions: torch.Tensor | None = None,
) -> ProjectedPredictorOutput:
    """Run one raw RGB-D window through the projected predictor."""
    model.warp_module.K.copy_(
        sample["intrinsics"].to(
            device=model.warp_module.K.device,
            dtype=model.warp_module.K.dtype,
        )
    )
    with autocast(DEVICE):
        return model(
            context_rgb=sample["context_rgb"],
            depth_t=sample["depth_t"],
            ego_motions=(
                sample["ego_motions"]
                if ego_motions is None
                else ego_motions
            ),
            future_rgb=None,
        )


def deterministic_proposal_rollout(
    model: ProjectedPredictor,
    output: ProjectedPredictorOutput,
    *,
    steps: int,
) -> torch.Tensor:
    """Construct the correction-free deterministic baseline ``[1,K,N,C]``.

    This intentionally does not use ``output.proposals``.  In the learned
    rollout, later entries of ``output.proposals`` receive the previous learned
    prediction on missing patches.  Here, the previous *baseline proposal* is
    fed back instead, preventing learned corrections from leaking across time.
    """
    z_previous = output.z_t.float()
    proposals: list[torch.Tensor] = []

    for step in range(steps):
        z_previous = model.latent_proposal(
            z_previous=z_previous,
            z_warped=output.warped_latents[:, step].float(),
            patch_valid=output.patch_valid[:, step],
        )
        proposals.append(z_previous)

    return torch.stack(proposals, dim=1)


def _cached_future_latents(
    dataset: KITTIProjectedPredictorDataset,
    sample: dict[str, Any],
    *,
    steps: int,
    device: torch.device | None,
) -> torch.Tensor:
    """Load cached target-frame V-JEPA latents as ``[1,K,N,C]``."""
    sequence_nr = _as_int(sample["sequence_nr"])
    start_index = _as_int(sample["start_index"])
    sequence = dataset.sequences[sequence_nr]

    latents: list[torch.Tensor] = []
    for step in range(steps):
        frame = _future_frame(dataset, start_index, step)
        try:
            array = sequence.get_vjepa_features(frame)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "the focused projected-predictor evaluator requires the cached "
                f"vjepa_vitb latent for sequence {sequence_nr:02d}, frame {frame}"
            ) from error

        # Copy protects against read-only numpy/memmap buffers before transfer.
        latent = torch.from_numpy(np.asarray(array).copy()).float()
        if latent.ndim == 3 and latent.shape[:2] == (GRID_HEIGHT, GRID_WIDTH):
            latent = latent.reshape(NUM_TOKENS, latent.shape[-1])
        if latent.ndim != 2 or latent.shape[0] != NUM_TOKENS:
            raise ValueError(
                f"cached latent for sequence {sequence_nr:02d}, frame {frame} "
                f"has shape {tuple(latent.shape)}; expected ({NUM_TOKENS}, C)"
            )
        latents.append(latent)

    stacked = torch.stack(latents, dim=0).unsqueeze(0)
    return stacked if device is None else stacked.to(device=device)


def to_chw(latent_nc: torch.Tensor) -> torch.Tensor:
    """Convert row-major ``[N,C]`` tokens to decoder input ``[1,C,H,W]``."""
    if latent_nc.ndim != 2 or latent_nc.shape[0] != NUM_TOKENS:
        raise ValueError(
            f"expected latent shape ({NUM_TOKENS}, C), got {tuple(latent_nc.shape)}"
        )
    return (
        latent_nc.reshape(GRID_HEIGHT, GRID_WIDTH, -1)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


# -----------------------------------------------------------------------------
# Latent evaluation
# -----------------------------------------------------------------------------


class FocusedLatentAccumulator:
    """Dataset-level per-horizon latent metrics for the final thesis tables."""

    KEYS = (
        "autoregressive_l1",
        "proposal_l1",
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
        proposal: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Add one batch of ``[B,K,N,C]`` latent sequences."""
        if not (real.shape == zero.shape == proposal.shape == target.shape):
            raise ValueError(
                "latent tensors must have identical shapes: "
                f"real={tuple(real.shape)}, zero={tuple(zero.shape)}, "
                f"proposal={tuple(proposal.shape)}, target={tuple(target.shape)}"
            )
        if real.ndim != 4 or real.shape[1] != self.steps:
            raise ValueError(
                f"expected [B,{self.steps},N,C] latents, got {tuple(real.shape)}"
            )

        values = {
            "autoregressive_l1": (
                real.float() - target.float()
            ).abs().mean(dim=(2, 3)),
            "proposal_l1": (
                proposal.float() - target.float()
            ).abs().mean(dim=(2, 3)),
            "zero_action_l1": (
                zero.float() - target.float()
            ).abs().mean(dim=(2, 3)),
            "action_sensitivity_l1": (
                real.float() - zero.float()
            ).abs().mean(dim=(2, 3)),
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
    model: ProjectedPredictor,
    loader: DataLoader,
    *,
    steps: int,
) -> dict[str, Any]:
    """Evaluate learned, deterministic, and zero-motion projected rollouts."""
    model.eval()
    accumulator = FocusedLatentAccumulator(steps)
    dataset = (
        loader.dataset.dataset
        if isinstance(loader.dataset, Subset)
        else loader.dataset
    )
    if not isinstance(dataset, KITTIProjectedPredictorDataset):
        raise TypeError("latent loader does not wrap KITTIProjectedPredictorDataset")

    start_time = time.time()
    for number, batch in enumerate(loader, start=1):
        sample = _unbatch(batch)
        real_output = run_model(model, sample)
        zero_predictions = run_model(
            model,
            sample,
            ego_motions=torch.zeros_like(sample["ego_motions"]),
        ).predictions.float()[:, :steps]

        target = _cached_future_latents(
            dataset,
            sample,
            steps=steps,
            device=DEVICE,
        )
        proposal = deterministic_proposal_rollout(
            model,
            real_output,
            steps=steps,
        )

        accumulator.update(
            real=real_output.predictions.float()[:, :steps],
            zero=zero_predictions,
            proposal=proposal,
            target=target,
        )

        if number % 100 == 0 or number == len(loader):
            elapsed = max(time.time() - start_time, 1e-9)
            rate = number / elapsed
            current = accumulator.summary()["autoregressive_l1"]
            print(
                f"  latent {number}/{len(loader)} windows "
                f"({rate:.2f} windows/s, AR L1 "
                f"{np.asarray(current).round(4).tolist()})",
                flush=True,
            )

    return accumulator.summary()


# -----------------------------------------------------------------------------
# Decoded evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_decoded(
    model: ProjectedPredictor,
    dataset: KITTIProjectedPredictorDataset,
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
    *,
    steps: int,
    every: int,
    max_windows: int | None,
) -> dict[str, Any]:
    """Decode predicted and cached target latents on the shared anchor grid."""
    model.eval()
    sources = ("predicted", "true")
    depth_accumulators = {
        source: [DepthMetricAccumulator() for _ in range(steps)]
        for source in sources
    }
    semantic_accumulators = {
        source: [SemanticMetricAccumulator() for _ in range(steps)]
        for source in sources
    }

    items = _anchor_items(dataset, every)
    if max_windows is not None:
        items = items[:max_windows]

    for number, item in enumerate(items, start=1):
        sample = dataset[item]
        sequence_nr = _as_int(sample["sequence_nr"])
        start_index = _as_int(sample["start_index"])
        sequence = dataset.sequences[sequence_nr]

        output = run_model(model, sample)
        predictions = output.predictions.float()[0, :steps]
        targets = _cached_future_latents(
            dataset,
            sample,
            steps=steps,
            device=DEVICE,
        )[0]

        for step in range(steps):
            frame = _future_frame(dataset, start_index, step)
            target_depth = sequence.get_depth(frame)
            target_semantics = sequence.get_semantics(frame)
            latents = {
                "predicted": predictions[step],
                "true": targets[step],
            }

            for source, latent in latents.items():
                decoded_depth, decoded_semantics = _decode(
                    depth_decoder,
                    semantic_decoder,
                    to_chw(latent),
                    target_depth.shape,
                    target_semantics.shape,
                )
                depth_accumulators[source][step].update(
                    target_depth,
                    decoded_depth,
                    target_semantics,
                )
                semantic_accumulators[source][step].update(
                    target_semantics,
                    decoded_semantics,
                )

        if number % 25 == 0 or number == len(items):
            print(f"  decoded {number}/{len(items)} windows", flush=True)

    depth_absrel = {
        source: [
            float(accumulator.summary()["non-sky"]["absrel"])
            for accumulator in accumulators
        ]
        for source, accumulators in depth_accumulators.items()
    }
    planning_miou = {
        source: [
            float(accumulator.summary()["planning_group_miou"])
            for accumulator in accumulators
        ]
        for source, accumulators in semantic_accumulators.items()
    }

    return {
        "windows": len(items),
        "every": every,
        "aggregation": (
            "one pixel-pooled accumulator per horizon and latent source "
            "over fixed-grid anchor windows"
        ),
        "depth_non_sky_absrel": depth_absrel["predicted"],
        "semantics_planning_group_miou": planning_miou["predicted"],
        "depth_non_sky_absrel_true": depth_absrel["true"],
        "semantics_planning_group_miou_true": planning_miou["true"],
    }


# -----------------------------------------------------------------------------
# Qualitative evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def collect_qualitative_example(
    model: ProjectedPredictor,
    spec: ModelSpec,
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
    *,
    steps: int,
) -> dict[str, Any]:
    """Collect one predicted rollout and paired pseudo-labels on CPU."""
    if spec.example_item is None:
        raise RuntimeError("qualitative example item has not been assigned")

    dataset = spec.dataset
    sample = dataset[spec.example_item]
    sequence_nr = _as_int(sample["sequence_nr"])
    start_index = _as_int(sample["start_index"])
    sequence = dataset.sequences[sequence_nr]

    output = run_model(model, sample)
    predictions = output.predictions.float()[0, :steps]

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

        decoded_depth, decoded_semantic = _decode(
            depth_decoder,
            semantic_decoder,
            to_chw(predictions[step]),
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


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def _horizon_label(value: float) -> str:
    return f"t+{value:.1f} s"


def summary_markdown(results: dict[str, dict[str, Any]]) -> str:
    """Create the focused thesis-facing Markdown tables."""
    lines = [
        "# Geometry-guided projected latent predictors -- held-out test set",
        "",
        "test sequences: "
        + ", ".join(f"{sequence:02d}" for sequence in TEST_SEQUENCES),
        "",
        "Latent L1 is the mean absolute difference over every spatial patch, "
        "all latent channels, and all evaluated test windows, reported "
        "separately at each rollout horizon. Targets are cached V-JEPA "
        "image latents of the actual future frames.",
        "",
        "The deterministic proposal baseline is a correction-free rollout: "
        "warp-valid patches take the RGB-D projected latent and missing "
        "patches copy the previous deterministic proposal. Learned predictions "
        "are never fed back into the baseline.",
        "",
    ]

    for tag, result in results.items():
        latent = result["latent"]
        horizons = result["horizon_seconds"]
        sample_note = (
            f"; latent sampling: every {result.get('latent_every', 1)}th window"
            if result.get("latent_every", 1) != 1
            else ""
        )
        geometry = result["geometry"]

        lines.extend(
            [
                f"## {result['step_seconds']:g} s-step model (`{tag}`)",
                "",
                f"checkpoint: `{result['checkpoint']}`  ",
                f"context: {result['context_length']} frames; "
                f"rollout: {result['horizon_steps']} steps; "
                f"latent windows: {latent['windows']}{sample_note}  ",
                f"geometry: splat radius {geometry['radius_px']:g} px; "
                f"patch-coverage threshold "
                f"{geometry['patch_coverage_threshold']:g}",
                "",
                "### Latent prediction error",
                "",
                markdown_table(
                    [
                        "Prediction horizon",
                        "Autoregressive latent L1 ↓",
                        "Deterministic-proposal latent L1 ↓",
                    ],
                    [
                        [
                            _horizon_label(horizon),
                            latent["autoregressive_l1"][index],
                            latent["proposal_l1"][index],
                        ]
                        for index, horizon in enumerate(horizons)
                    ],
                ),
                "",
            ]
        )

        decoded = result.get("decoded")
        if decoded is not None:
            absrel_true = decoded.get("depth_non_sky_absrel_true")
            miou_true = decoded.get("semantics_planning_group_miou_true")
            lines.extend(
                [
                    "### Decoded performance: depth",
                    "",
                    markdown_table(
                        [
                            "Prediction horizon",
                            "Cached-latent non-sky AbsRel ↓",
                            "Predicted-latent non-sky AbsRel ↓",
                        ],
                        [
                            [
                                _horizon_label(horizon),
                                absrel_true[index] if absrel_true else None,
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
                            "Cached-latent planning-group mIoU ↑",
                            "Predicted-latent planning-group mIoU ↑",
                        ],
                        [
                            [
                                _horizon_label(horizon),
                                miou_true[index] if miou_true else None,
                                decoded["semantics_planning_group_miou"][index],
                            ]
                            for index, horizon in enumerate(horizons)
                        ],
                    ),
                    "",
                    f"Decoded metrics use {decoded['windows']} fixed-grid test "
                    f"windows (every {decoded['every']} raw frames) and are "
                    "pooled over pixels separately at each horizon. "
                    "Cached-latent columns decode the frozen V-JEPA encoding "
                    "of the actual target frame; predicted-latent columns "
                    "decode the projected predictor's learned rollout.",
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
                "`Zero − real L1` is positive when the recorded ego motion "
                "improves target prediction. `Action sensitivity` is the "
                "direct mean absolute difference between the real-motion and "
                "zero-motion predicted latents.",
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
    """Print the same core quantities in a compact console table."""
    latent = result["latent"]
    decoded = result.get("decoded")

    print(f"\n[{tag}] {result['step_seconds']:g} s-step model")
    print(
        f"{'horizon':<10} {'AR L1':>9} {'proposal':>10} "
        f"{'zero L1':>9} {'zero-real':>10} {'sensitivity':>12}"
    )
    for index, horizon in enumerate(result["horizon_seconds"]):
        print(
            f"{_horizon_label(horizon):<10} "
            f"{fmt(latent['autoregressive_l1'][index]):>9} "
            f"{fmt(latent['proposal_l1'][index]):>10} "
            f"{fmt(latent['zero_action_l1'][index]):>9} "
            f"{fmt(latent['zero_minus_real_l1'][index]):>10} "
            f"{fmt(latent['action_sensitivity_l1'][index]):>12}"
        )

    if decoded:
        absrel_true = decoded.get("depth_non_sky_absrel_true")
        miou_true = decoded.get("semantics_planning_group_miou_true")
        print("\nDecoded metrics (cached target latent vs predicted latent)")
        print(
            f"{'horizon':<10} {'AbsRel cached':>14} {'AbsRel pred':>12} "
            f"{'mIoU cached':>12} {'mIoU pred':>10}"
        )
        for index, horizon in enumerate(result["horizon_seconds"]):
            print(
                f"{_horizon_label(horizon):<10} "
                f"{fmt(absrel_true[index] if absrel_true else None):>14} "
                f"{fmt(decoded['depth_non_sky_absrel'][index]):>12} "
                f"{fmt(miou_true[index] if miou_true else None):>12} "
                f"{fmt(decoded['semantics_planning_group_miou'][index]):>10}"
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
    """Load one model, run the focused evaluation, then release it."""
    model, checkpoint = load_projected_predictor(
        spec.path,
        vjepa_checkpoint=args.vjepa_checkpoint,
        radius_px=args.radius_px,
        patch_coverage_threshold=args.patch_coverage_threshold,
    )
    checkpoint_line = describe_checkpoint(spec.path, checkpoint)

    loaded_step = float(checkpoint["step_seconds"])
    if not math.isclose(loaded_step, spec.step_seconds, abs_tol=1e-8):
        raise ValueError(
            f"{spec.path}: metadata changed between preparation and load "
            f"({spec.step_seconds}s vs {loaded_step}s)"
        )

    print(
        f"\nloaded {checkpoint_line}\n"
        f"  V-JEPA weights: {checkpoint['vjepa_checkpoint_used']}\n"
        f"[{spec.tag}] {len(spec.dataset)} available test windows; "
        f"{args.rollout_steps} steps of {spec.step_seconds:g}s"
    )

    latent_items = list(range(0, len(spec.dataset), args.latent_every))
    if args.max_windows is not None:
        latent_items = latent_items[:args.max_windows]
    latent_loader = DataLoader(
        Subset(spec.dataset, latent_items),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    print(f"[{spec.tag}] scoring {len(latent_items)} latent windows ...")
    latent = evaluate_latent(
        model,
        latent_loader,
        steps=args.rollout_steps,
    )

    result: dict[str, Any] = {
        "checkpoint": checkpoint_line,
        "checkpoint_path": str(spec.path),
        "vjepa_checkpoint": checkpoint["vjepa_checkpoint_used"],
        "iteration": checkpoint.get("iteration"),
        "val": checkpoint.get("val"),
        "step_seconds": spec.step_seconds,
        "frame_stride": spec.frame_stride,
        "context_length": CONTEXT_LENGTH,
        "horizon_steps": args.rollout_steps,
        "horizon_seconds": [
            (step + 1) * spec.step_seconds
            for step in range(args.rollout_steps)
        ],
        "test_sequences": TEST_SEQUENCES,
        "target_latent_source": "cached vjepa_vitb encoding of each future frame",
        "baseline": {
            "name": "deterministic latent proposal",
            "definition": (
                "warp latent on valid patches; previous deterministic proposal "
                "on missing patches; no learned correction feedback"
            ),
        },
        "geometry": {
            "radius_px": args.radius_px,
            "patch_coverage_threshold": args.patch_coverage_threshold,
            "image_size": list(IMAGE_SIZE),
            "patch_size": PATCH_SIZE,
        },
        "latent_every": args.latent_every,
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


def _write_combined_outputs(
    results: dict[str, dict[str, Any]],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(
        {
            "evaluation": "focused geometry-guided projected predictor",
            "models": results,
        },
        figures_dir / "metrics.json",
    )
    (figures_dir / "summary.md").write_text(summary_markdown(results))

    for tag, result in results.items():
        if result.get("decoded"):
            plot_decoded_curves(
                tag,
                result,
                figures_dir / f"decoded_metrics_vs_horizon_{tag}.png",
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused evaluation of geometry-guided projected predictors"
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "projected-predictor checkpoints; by default evaluates every "
            "checkpoints_projected_predictor/projected_predictor_dt*.pt"
        ),
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=MAX_STEPS,
        choices=range(1, MAX_STEPS + 1),
        help=(
            f"number of horizons to score; the architecture always computes "
            f"{MAX_STEPS} and this option may select an initial prefix"
        ),
    )
    parser.add_argument(
        "--latent-every",
        type=int,
        default=1,
        help="score every Nth available test window in latent space",
    )
    parser.add_argument(
        "--decoded-every",
        type=int,
        default=20,
        help="decode windows whose final context frame lies every N raw frames",
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
        help="do not generate the three qualitative rollout figures",
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
        "--vjepa-checkpoint",
        type=Path,
        default=None,
        help=(
            "released V-JEPA 2.1 ViT-B checkpoint; defaults to the checkpoint "
            "recorded by training, then the repository default"
        ),
    )
    parser.add_argument(
        "--radius-px",
        type=float,
        default=DEFAULT_RADIUS_PX,
        help="warp splat radius; must match training",
    )
    parser.add_argument(
        "--patch-coverage-threshold",
        type=float,
        default=DEFAULT_PATCH_COVERAGE_THRESHOLD,
        help="warp patch-validity threshold; must match training",
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
            "skip evaluation and rebuild summary.md, combined metrics.json, "
            "and decoded curves from compatible per-model metrics.json files"
        ),
    )
    args = parser.parse_args()

    if args.latent_every <= 0:
        raise SystemExit("--latent-every must be positive")
    if args.decoded_every <= 0:
        raise SystemExit("--decoded-every must be positive")
    if args.num_workers < 0:
        raise SystemExit("--num-workers cannot be negative")
    if args.max_windows is not None and args.max_windows <= 0:
        raise SystemExit("--max-windows must be positive")
    if args.radius_px <= 0:
        raise SystemExit("--radius-px must be positive")
    if not 0 <= args.patch_coverage_threshold <= 1:
        raise SystemExit("--patch-coverage-threshold must lie in [0, 1]")

    if args.replot:
        results = {
            path.parent.name: json.loads(path.read_text())
            for path in sorted(args.figures_dir.glob("*/metrics.json"))
        }
        results = {
            tag: result
            for tag, result in results.items()
            if "proposal_l1" in result.get("latent", {})
        }
        if not results:
            raise SystemExit(
                f"no compatible <tag>/metrics.json under "
                f"{args.figures_dir} to replot"
            )

        print("replotting from: " + ", ".join(results))
        for tag, result in results.items():
            print_overview(tag, result)
        _write_combined_outputs(results, args.figures_dir)
        print(f"combined metrics:   {args.figures_dir / 'metrics.json'}")
        print(f"thesis tables:      {args.figures_dir / 'summary.md'}")
        return

    checkpoint_paths = list(args.checkpoints or default_checkpoints())
    if not checkpoint_paths:
        raise SystemExit(
            "no projected-predictor checkpoints found in "
            f"{PROJECTED_CHECKPOINT_DIR}"
        )

    required_paths = checkpoint_paths + [
        args.depth_checkpoint,
        args.semantics_checkpoint,
    ]
    if args.vjepa_checkpoint is not None:
        required_paths.append(args.vjepa_checkpoint)
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(
            "required files do not exist: "
            + ", ".join(str(path) for path in missing)
        )

    specs = prepare_specs(checkpoint_paths)

    if not args.skip_figures:
        sequence, anchor = assign_shared_example(
            specs,
            requested_sequence=args.example_sequence,
            requested_anchor=args.example_anchor,
        )
        print(
            "qualitative example shared across projected models: "
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

    _write_combined_outputs(results, args.figures_dir)

    print()
    for example in qualitative_examples:
        tag = example["tag"]
        depth_path = plot_depth_rollout(
            example,
            args.figures_dir / f"depth_rollout_{tag}.png",
        )
        fine_path = plot_semantic_rollout(
            example,
            args.figures_dir / f"semantics_rollout_{tag}.png",
            planning=False,
        )
        planning_path = plot_semantic_rollout(
            example,
            args.figures_dir / f"planning_semantics_rollout_{tag}.png",
            planning=True,
        )
        print(f"[{tag}] depth figure:              {depth_path}")
        print(f"[{tag}] fine semantics figure:     {fine_path}")
        print(f"[{tag}] planning semantics figure: {planning_path}")

    for tag, result in results.items():
        if result.get("decoded"):
            print(
                f"[{tag}] decoded curves:            "
                f"{args.figures_dir / f'decoded_metrics_vs_horizon_{tag}.png'}"
            )

    print(f"combined metrics:   {args.figures_dir / 'metrics.json'}")
    print(f"thesis tables:      {args.figures_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
