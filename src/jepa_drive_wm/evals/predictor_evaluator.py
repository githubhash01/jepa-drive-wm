"""
Evaluate the trained ac-style world models on the held-out KITTI test sequences
(SPLIT_V1.test_sequences, see data/splits.py).

    PYTHONPATH=src python -m jepa_drive_wm.evals.predictor_evaluator \
        [--checkpoints outputs/checkpoints_wm/world_model_dt0.2s.pt ...] [--rollout-steps 4]

By default every `world_model_dt*.pt` in OUTPUTS_DIR/checkpoints_wm is
evaluated. Each checkpoint records its own step duration (`step_seconds`), so
the frame stride of the test windows is derived per model (0.2 s -> stride 2,
0.5 s -> stride 5). Every model is rolled out for --rollout-steps steps
(default 4: 0.8 s for the 0.2 s model, 2.0 s for the 0.5 s model); pass
--horizon-seconds instead to roll every model to one physical horizon. All
curves are plotted against horizon in seconds either way.

Per model, over every test window:

- latent-space metrics: mean L1 in the frozen V-JEPA 2.1 space per rollout step, for
      rollout              autoregressive (own predictions fed back as context)
      teacher_forced       one-step prediction from the true history
      copy_last            copy the last context frame -- the model's own
                           initialisation prior, so beating it is exactly
                           "the model learned dynamics"
      copy_previous        copy the true frame one step back -- the matching
                           baseline for teacher_forced, whose window already
                           holds the true frames up to the previous step
      rollout_zero_action  rollout with the ego motion zeroed -- if this is as
                           good as `rollout`, the model ignores its actions
  overall and per test sequence (01 highway / 07 urban / 09 rural);
- decoded-task metrics on a subsample of windows (--decoded-every, a frame grid
  of anchor frames shared by all models, so every model is scored on exactly
  the same frames at common horizons): the
  predicted latents are pushed through the frozen depth and semantic decoders
  and scored against the pseudolabels with exactly the code the decoder
  evaluators use (DepthMetricAccumulator / SemanticMetricAccumulator), next to
  the same decoders run on the true latent (the decoders' ceiling) and on the
  copy-last-frame latent (the trivial baseline) -- world-model error isolated
  from decoder error, in units that matter for driving;
- figures: for a few windows, the rollout decoded to depth and semantics: a
  first column for the last context frame t (camera I_t, pseudolabel GT at t,
  decoders on z_t -- what the rollout starts from), then one column per step
  (camera, pseudolabel GT, decoder on true latent, on copy latent, on
  predicted latent).

Outputs in OUTPUTS_DIR/evals_wm:
    <tag>/metrics.json                 everything for one model (tag = dt0.2s, dt0.5s for the
                                       trainer's default filenames; other files get the
                                       checkpoint stem appended so same-step runs never collide)
    <tag>/seqXX_windowNNNNNN_{depth,semantics}.png
    latent_l1_vs_horizon.png           all models: rollout / TF / copy / zero-action L1 vs horizon
    decoded_metrics_vs_horizon.png     all models: depth AbsRel, delta1, mIoU, planning mIoU vs horizon
    summary.md                         tables for all models
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from jepa_drive_wm.data.data_interface_rollout import KITTIRolloutDataset
from jepa_drive_wm.data.splits import SPLIT_V1
from jepa_drive_wm.evals.common import (
    DEPTH_CHECKPOINT,
    DEVICE,
    MUTED,
    SEMANTICS_CHECKPOINT,
    SERIES,
    WM_CHECKPOINT_DIR,
    describe_checkpoint,
    finish_figure,
    load_depth_decoder,
    load_semantic_decoder,
    load_world_model,
    markdown_table,
    new_figure,
    shared_legend_below,
    style_axes,
    write_metrics_json,
)
from jepa_drive_wm.evals.depth_evaluator import DepthMetricAccumulator
from jepa_drive_wm.evals.semantics_evaluator import SemanticMetricAccumulator
from jepa_drive_wm.models.dense_decoders.depth_decoder import DepthDecoder
from jepa_drive_wm.models.dense_decoders.semantic_decoder import SemanticDecoder
from jepa_drive_wm.models.predictors.ac_style.ac_predictor import VJEPA21WorldModel
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.train.train_ac_predictor import MAX_SPEED_MPS, action_scale, forward_predictions, to_grid
from jepa_drive_wm.train.train_depth import MAX_DEPTH, MIN_DEPTH, predict as predict_depth
from jepa_drive_wm.train.train_semantics import predict as predict_semantics
from jepa_drive_wm.training_utils import autocast
from jepa_drive_wm.viz.visualiser import class_colors

FIGURES_DIR = OUTPUTS_DIR / "evals_wm"
TEST_SEQUENCES = list(SPLIT_V1.test_sequences)
FRAME_PERIOD = KITTIRolloutDataset.FRAME_PERIOD  # KITTI odometry is 10 Hz

# Latent-space variants, in display order.
VARIANTS = {
    "rollout": "autoregressive rollout",
    "teacher_forced": "one-step (teacher forced)",
    "copy_last": "copy last context frame",
    "copy_previous": "copy previous true frame",
    "rollout_zero_action": "rollout, zero action",
}
# Latent sources fed to the frozen decoders, in display order.
SOURCES = {
    "true": "true latent (decoder ceiling)",
    "copy": "copy-last-frame latent",
    "predicted": "predicted latent (rollout)",
}
# At most this many rollout steps are shown as columns in the qualitative figures.
MAX_FIGURE_COLUMNS = 5
# Qualitative figures pick their windows from anchors on this frame grid, so two
# models' figures show the same scenes.
FIGURE_ANCHOR_GRID = 50


def default_checkpoints() -> list[Path]:
    """Every horizon-tagged world-model checkpoint the trainer can have written."""
    return sorted(WM_CHECKPOINT_DIR.glob("world_model_dt*.pt"))


def model_tag(checkpoint: dict, path: Path) -> str:
    """
    Names the run, its results entry and its output folder. The trainer's
    default filename world_model_<step tag>.pt gets the bare step tag (dt0.2s,
    dt0.5s); any other file keeps its stem too, so two runs at the same step
    (old vs retrained) never overwrite each other.
    """
    tag = f"dt{checkpoint['step_seconds']:g}s"
    return tag if path.stem == f"world_model_{tag}" else f"{tag}_{path.stem}"


def frame_stride_for(step_seconds: float) -> int:
    """Invert KITTIRolloutDataset.step_seconds = frame_stride * FRAME_PERIOD."""
    stride = step_seconds / FRAME_PERIOD
    if not math.isclose(stride, round(stride), abs_tol=1e-6):
        raise ValueError(f"step {step_seconds}s is not a whole number of {FRAME_PERIOD}s frames")
    return int(round(stride))


# ----------------------------------------------------------------------------- latent metrics

class LatentMetrics:
    """Per-step mean latent L1 for each variant, pooled over windows; also per sequence."""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self._sum: dict[str, np.ndarray] = {}
        self._seq_sum: dict[str, dict[int, np.ndarray]] = {}
        self._windows = 0
        self._seq_windows: dict[int, int] = {}

    def update(self, variant: str, per_window_l1: torch.Tensor, sequence_nrs: torch.Tensor) -> None:
        """per_window_l1: [B, K] mean |pred - target| per window and step (fp32, cpu)."""
        values = per_window_l1.double().numpy()
        self._sum[variant] = self._sum.get(variant, np.zeros(self.steps)) + values.sum(axis=0)
        seq_sum = self._seq_sum.setdefault(variant, {})
        for row, seq in zip(values, sequence_nrs.tolist()):
            seq_sum[seq] = seq_sum.get(seq, np.zeros(self.steps)) + row
        if variant == "rollout":  # count windows once, not once per variant
            self._windows += values.shape[0]
            for seq in sequence_nrs.tolist():
                self._seq_windows[seq] = self._seq_windows.get(seq, 0) + 1

    @property
    def windows(self) -> int:
        return self._windows

    def mean(self, variant: str) -> np.ndarray:
        return self._sum[variant] / max(self._windows, 1)

    def per_sequence(self, variant: str) -> dict[int, np.ndarray]:
        return {seq: self._seq_sum[variant][seq] / self._seq_windows[seq]
                for seq in sorted(self._seq_windows)}

    def summary(self) -> dict:
        return {
            "windows": self._windows,
            "windows_per_sequence": {f"{s:02d}": n for s, n in sorted(self._seq_windows.items())},
            "variants": {
                variant: {
                    "l1": self.mean(variant).tolist(),
                    "l1_mean_over_steps": float(self.mean(variant).mean()),
                    "l1_per_sequence": {f"{s:02d}": v.tolist() for s, v in self.per_sequence(variant).items()},
                }
                for variant in VARIANTS if variant in self._sum
            },
        }


@torch.no_grad()
def evaluate_latent(model: VJEPA21WorldModel, loader, act_scale: torch.Tensor, steps: int) -> LatentMetrics:
    """One pass over the test windows: per-step L1 for every variant in VARIANTS."""
    model.eval()
    metrics = LatentMetrics(steps)
    H, W = model.grid_height, model.grid_width

    for batch in loader:
        context = to_grid(batch["context_latents"].to(DEVICE, non_blocking=True), H, W)
        future = to_grid(batch["future_latents"].to(DEVICE, non_blocking=True), H, W)
        ego_motions = batch["future_ego_motions"].to(DEVICE) / act_scale
        sequence_nrs = batch["sequence_nr"]

        with autocast(DEVICE):
            tf_preds, ar_preds = forward_predictions(model, context, ego_motions, future)
            zero_preds = model.rollout(context, torch.zeros_like(ego_motions))

        def per_window_l1(pred: torch.Tensor) -> torch.Tensor:
            # mean over (H, W, C) in fp32, like latent_loss, but kept per window and step
            return (pred.float() - future).abs().mean(dim=(2, 3, 4)).cpu()

        metrics.update("rollout", per_window_l1(ar_preds), sequence_nrs)
        metrics.update("teacher_forced", per_window_l1(tf_preds), sequence_nrs)
        metrics.update("copy_last", per_window_l1(context[:, -1:].expand_as(future)), sequence_nrs)
        # one-step baseline: the true frame one step back (context[-1] for step 0)
        metrics.update("copy_previous", per_window_l1(torch.cat([context[:, -1:], future[:, :-1]], dim=1)),
                       sequence_nrs)
        metrics.update("rollout_zero_action", per_window_l1(zero_preds), sequence_nrs)
    return metrics


# ----------------------------------------------------------------------------- decoded metrics

def _future_frame(dataset: KITTIRolloutDataset, start_index: int, k: int) -> int:
    """Raw frame index of rollout step k (0-based) for a window starting at start_index."""
    return start_index + (dataset.context_length + k) * dataset.frame_stride


def _anchor_items(dataset: KITTIRolloutDataset, every: int) -> list[int]:
    """
    Dataset items whose last context frame (the "anchor") lies on a fixed
    per-sequence frame grid: anchors 0, every, 2*every, ... Step k of such a
    window targets frame anchor + (k + 1) * frame_stride = anchor + 10 * horizon,
    so models with different step durations score exactly the same frames at
    common horizons. (Subsampling by window *position* would hand each model a
    different frame subset, since the window span depends on the stride.)
    """
    lookup = {key: item for item, key in enumerate(dataset.index)}
    lead = (dataset.context_length - 1) * dataset.frame_stride
    return [lookup[(sequence_nr, anchor - lead)]
            for sequence_nr, sequence in dataset.sequences.items()
            for anchor in range(0, len(sequence), every)
            if (sequence_nr, anchor - lead) in lookup]


@torch.no_grad()
def _decode(depth_decoder, semantic_decoder, latent_chw: torch.Tensor,
            depth_shape: tuple[int, int], semantics_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """[1, C, H, W] latent -> (metric depth (H, W), class ids (H, W)) at pseudolabel resolution."""
    depth = predict_depth(depth_decoder, latent_chw, depth_shape).cpu().numpy()
    semantics = predict_semantics(semantic_decoder, latent_chw, semantics_shape).argmax(dim=1)[0]
    return depth, semantics.to(torch.uint8).cpu().numpy()


@torch.no_grad()
def evaluate_decoded(
    model: VJEPA21WorldModel,
    dataset: KITTIRolloutDataset,
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
    act_scale: torch.Tensor,
    steps: int,
    every: int,
    max_windows: int | None = None,
) -> dict:
    """
    Windows anchored on every `every`-th frame of each test sequence (see
    _anchor_items): roll the model out, decode true / copy / predicted latents
    at each step with the frozen decoders and score them against the
    pseudolabels of that step's frame.
    """
    model.eval()
    H, W = model.grid_height, model.grid_width
    depth_acc = {source: [DepthMetricAccumulator() for _ in range(steps)] for source in SOURCES}
    sem_acc = {source: [SemanticMetricAccumulator() for _ in range(steps)] for source in SOURCES}

    items = _anchor_items(dataset, every)
    if max_windows is not None:
        items = items[:max_windows]

    for n, item in enumerate(items):
        sample = dataset[item]
        sequence = dataset.sequences[sample["sequence_nr"].item()]
        start_index = sample["start_index"].item()

        context = to_grid(sample["context_latents"][None].to(DEVICE), H, W)
        ego_motions = sample["future_ego_motions"][None].to(DEVICE) / act_scale
        with autocast(DEVICE):
            preds = model.rollout(context, ego_motions)[0].float()        # [K, H, W, C]
        copy_chw = context[0, -1].permute(2, 0, 1)[None]                   # [1, C, H, W]

        for k in range(steps):
            frame = _future_frame(dataset, start_index, k)
            gt_depth = sequence.get_depth(frame)
            gt_semantics = sequence.get_semantics(frame)
            latents = {
                "true": sample["future_latents"][k].view(H, W, -1).permute(2, 0, 1)[None].to(DEVICE),
                "copy": copy_chw,
                "predicted": preds[k].permute(2, 0, 1)[None],
            }
            for source, latent in latents.items():
                depth, semantics = _decode(depth_decoder, semantic_decoder, latent,
                                           gt_depth.shape, gt_semantics.shape)
                depth_acc[source][k].update(gt_depth, depth, gt_semantics)
                sem_acc[source][k].update(gt_semantics, semantics)
        if (n + 1) % 25 == 0 or n + 1 == len(items):
            print(f"  decoded {n + 1}/{len(items)} windows", flush=True)

    return {
        "windows": len(items),
        "every": every,
        "anchor_frames": "last context frame on the grid 0, every, 2*every, ... of each sequence",
        "depth": {source: [acc.summary() for acc in accs] for source, accs in depth_acc.items()},
        "semantics": {source: [acc.summary() for acc in accs] for source, accs in sem_acc.items()},
    }


# ----------------------------------------------------------------------------- qualitative

def _save_grid(rows, column_titles: list[str], title: str, path: Path) -> None:
    """One figure: rows = (label, [image per column], imshow kwargs), columns = context t + steps."""
    n_cols = len(column_titles)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(6 * n_cols, 2.1 * len(rows)), squeeze=False)
    for row, (label, images, kwargs) in enumerate(rows):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.imshow(images[col], **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(column_titles[col])
            if col == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def save_figures(
    model: VJEPA21WorldModel,
    dataset: KITTIRolloutDataset,
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
    act_scale: torch.Tensor,
    steps: int,
    n_rollouts: int,
    figures_dir: Path,
) -> None:
    """
    For n_rollouts test windows evenly spaced over the FIGURE_ANCHOR_GRID
    anchors (so every model shows the same scenes): one depth figure and one
    semantics figure. Columns: the last context frame t (camera I_t, pseudolabel
    GT at t, the decoders on the true latent z_t -- identical in the three
    decoder rows, since both the copy baseline and the rollout start from z_t),
    then up to MAX_FIGURE_COLUMNS rollout steps. Rows: camera, pseudolabel GT,
    decoder on true latent, on copy-last latent, on predicted latent.
    """
    H, W = model.grid_height, model.grid_width
    shown = sorted(set(np.linspace(0, steps - 1, min(steps, MAX_FIGURE_COLUMNS)).round().astype(int)))
    column_titles = ["t (last context)"] + [f"t + {(k + 1) * dataset.step_seconds:.1f}s" for k in shown]
    candidates = _anchor_items(dataset, FIGURE_ANCHOR_GRID)
    picks = sorted(set(np.linspace(0, len(candidates) - 1, min(n_rollouts, len(candidates))).round().astype(int)))

    for item in (candidates[i] for i in picks):
        sample = dataset[item]
        sequence_nr = sample["sequence_nr"].item()
        start_index = sample["start_index"].item()
        sequence = dataset.sequences[sequence_nr]

        context = to_grid(sample["context_latents"][None].to(DEVICE), H, W)
        ego_motions = sample["future_ego_motions"][None].to(DEVICE) / act_scale
        with autocast(DEVICE):
            preds = model.rollout(context, ego_motions)[0].float()  # [K, H, W, C]
        copy_chw = context[0, -1].permute(2, 0, 1)[None]

        # column 0: the last context frame t, everything decoded from z_t
        anchor = _future_frame(dataset, start_index, -1)
        gt_depth_t = sequence.get_depth(anchor)
        gt_semantics_t = sequence.get_semantics(anchor)
        depth_t, semantics_t = _decode(depth_decoder, semantic_decoder, copy_chw,
                                       gt_depth_t.shape, gt_semantics_t.shape)
        cameras, depth_gt, sem_gt = [sequence.get_image(anchor)], [gt_depth_t], [class_colors(gt_semantics_t)]
        depth_by = {source: [depth_t] for source in SOURCES}
        sem_by = {source: [class_colors(semantics_t)] for source in SOURCES}
        for k in shown:
            frame = _future_frame(dataset, start_index, k)
            gt_depth = sequence.get_depth(frame)
            gt_semantics = sequence.get_semantics(frame)
            cameras.append(sequence.get_image(frame))
            depth_gt.append(gt_depth)
            sem_gt.append(class_colors(gt_semantics))
            latents = {
                "true": sample["future_latents"][k].view(H, W, -1).permute(2, 0, 1)[None].to(DEVICE),
                "copy": copy_chw,
                "predicted": preds[k].permute(2, 0, 1)[None],
            }
            for source, latent in latents.items():
                depth, semantics = _decode(depth_decoder, semantic_decoder, latent,
                                           gt_depth.shape, gt_semantics.shape)
                depth_by[source].append(depth)
                sem_by[source].append(class_colors(semantics))

        depth_kwargs = dict(cmap="plasma", vmin=MIN_DEPTH, vmax=MAX_DEPTH)
        stem = f"seq{sequence_nr:02d}_window{start_index:06d}"
        title = (f"seq {sequence_nr:02d}, last context frame t = {anchor} "
                 f"(window from frame {start_index}, step {dataset.step_seconds:g}s)")
        _save_grid(
            [
                ("camera", cameras, {}),
                ("FoundationStereo\ndepth", depth_gt, depth_kwargs),
                ("depth from\ntrue latent", depth_by["true"], depth_kwargs),
                ("depth from\ncopy-last latent", depth_by["copy"], depth_kwargs),
                ("depth from\npredicted latent", depth_by["predicted"], depth_kwargs),
            ],
            column_titles, title, figures_dir / f"{stem}_depth.png",
        )
        _save_grid(
            [
                ("camera", cameras, {}),
                ("OneFormer\nsemantics", sem_gt, {}),
                ("semantics from\ntrue latent", sem_by["true"], {}),
                ("semantics from\ncopy-last latent", sem_by["copy"], {}),
                ("semantics from\npredicted latent", sem_by["predicted"], {}),
            ],
            column_titles, title, figures_dir / f"{stem}_semantics.png",
        )


# ----------------------------------------------------------------------------- comparison plots

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


def plot_latent_curves(results: dict[str, dict], path: Path) -> None:
    """
    Left: latent L1 vs horizon for every model (colour) and variant (line style).
    Right: each prediction relative to its matching copy baseline (1.0 = no
    better than copying; below = learned dynamics): the rollouts against
    copy-last (same context), teacher forcing against copy-previous (its window
    already holds the true frames up to the previous step).
    """
    fig, axes = new_figure(ncols=2, width=6.4, height=4.4)
    ax_abs, ax_rel = axes[0]
    baseline_of = {"rollout": "copy_last", "rollout_zero_action": "copy_last", "teacher_forced": "copy_previous"}
    for i, (tag, result) in enumerate(results.items()):
        color = SERIES[i % len(SERIES)]
        horizons = np.asarray(result["horizon_seconds"])
        variants = result["latent"]["variants"]
        for variant, style in VARIANT_STYLE.items():
            if variant not in variants:
                continue
            values = np.asarray(variants[variant]["l1"])
            ax_abs.plot(horizons, values, color=color, label=f"{tag} - {VARIANTS[variant]}", **style)
            if variant in baseline_of and baseline_of[variant] in variants:
                baseline = np.asarray(variants[baseline_of[variant]]["l1"])
                ax_rel.plot(horizons, values / baseline, color=color,
                            label=f"{tag} - {VARIANTS[variant]}", **style)
    ax_rel.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=":")
    ax_rel.text(ax_rel.get_xlim()[1], 1.0, " copy baseline", va="center", ha="right", fontsize=8, color=MUTED)
    ax_rel.text(0.02, 0.03, "rollouts / copy-last;  one-step / copy-previous", transform=ax_rel.transAxes,
                fontsize=8, color=MUTED, ha="left", va="bottom")
    style_axes(ax_abs, title="Latent L1 vs horizon (test set)", xlabel="horizon (s)",
               ylabel="mean |pred - true| in V-JEPA space")
    style_axes(ax_rel, title="Relative to the matching copy baseline", xlabel="horizon (s)",
               ylabel="L1 / copy-baseline L1")
    # colour = model, line style = variant; one legend for both panels
    bottom = shared_legend_below(fig, ax_abs, ncol=4)
    finish_figure(fig, path, rect=(0, bottom, 1, 1))


def plot_decoded_curves(results: dict[str, dict], path: Path) -> None:
    """2x2 small multiples: depth AbsRel / delta1 (non-sky), semantic mIoU / planning mIoU vs horizon."""
    panels = [
        ("depth", "non-sky", "absrel", "Depth AbsRel (non-sky), lower is better"),
        ("depth", "non-sky", "d1", "Depth delta1 (non-sky), higher is better"),
        ("semantics", None, "miou", "Semantic mIoU (19-class), higher is better"),
        ("semantics", None, "planning_group_miou", "Planning-group mIoU, higher is better"),
    ]
    fig, axes = new_figure(ncols=2, nrows=2, width=6.4, height=4.2)
    for ax, (task, scope, key, title) in zip(axes.flat, panels):
        for i, (tag, result) in enumerate(results.items()):
            decoded = result.get("decoded")
            if not decoded:
                continue
            color = SERIES[i % len(SERIES)]
            horizons = np.asarray(result["horizon_seconds"])
            for source, style in SOURCE_STYLE.items():
                per_step = decoded[task][source]
                values = [(m[scope][key] if scope else m[key]) for m in per_step]
                ax.plot(horizons, values, color=color, label=f"{tag} - {SOURCES[source]}", **style)
        style_axes(ax, title=title, xlabel="horizon (s)", ylabel=key)
    # colour = model, line style = latent source; one legend for all four panels
    bottom = shared_legend_below(fig, axes[0, 0], ncol=3)
    finish_figure(fig, path, rect=(0, bottom, 1, 1))


# ----------------------------------------------------------------------------- reporting

def summary_markdown(results: dict[str, dict]) -> str:
    lines = [
        "# ac-style world models -- test set (SPLIT_V1: sequences "
        + ", ".join(f"{s:02d}" for s in TEST_SEQUENCES) + ")",
        "",
        "Latent L1 = mean |pred - true| in the frozen V-JEPA 2.1 space, pooled over all test windows "
        "(same quantity as the trainer's tf/ar losses). Decoded metrics push latents through the "
        "frozen depth / semantic decoders and score them against the pseudolabels with the decoder "
        "evaluators' code.",
        "",
    ]
    for tag, result in results.items():
        horizons = result["horizon_seconds"]
        variants = result["latent"]["variants"]
        lines += [
            f"## {tag}",
            "",
            f"checkpoint: `{result['checkpoint']}`  ",
            f"context {result['context_length']} frames, frame stride {result['frame_stride']}, "
            f"{result['horizon_steps']} rollout steps of {result['step_seconds']:g}s, "
            f"{result['latent']['windows']} test windows",
            "",
            "### Latent L1 per step",
            "",
            markdown_table(
                ["horizon (s)", *[VARIANTS[v] for v in VARIANTS if v in variants], "rollout / copy"],
                [[f"+{h:.1f}", *[variants[v]["l1"][k] for v in VARIANTS if v in variants],
                  variants["rollout"]["l1"][k] / variants["copy_last"]["l1"][k]]
                 for k, h in enumerate(horizons)],
            ),
            "",
            "### Rollout L1 per sequence",
            "",
            markdown_table(
                ["seq", "windows", *[f"rollout +{h:.1f}s" for h in horizons],
                 *[f"copy +{h:.1f}s" for h in (horizons[0], horizons[-1])]],
                [[seq, result["latent"]["windows_per_sequence"][seq],
                  *variants["rollout"]["l1_per_sequence"][seq],
                  variants["copy_last"]["l1_per_sequence"][seq][0],
                  variants["copy_last"]["l1_per_sequence"][seq][-1]]
                 for seq in result["latent"]["windows_per_sequence"]],
            ),
            "",
        ]
        decoded = result.get("decoded")
        if decoded:
            rows = []
            for k, h in enumerate(horizons):
                for source in SOURCES:
                    d = decoded["depth"][source][k]
                    s = decoded["semantics"][source][k]
                    rows.append([f"+{h:.1f}  {SOURCES[source]}", d["non-sky"]["absrel"], d["non-sky"]["rmse"],
                                 d["non-sky"]["d1"], d["vehicle"]["absrel"], s["miou"],
                                 s["planning_group_miou"], s["drivable_iou"], s["drivable_boundary_iou"],
                                 s["traffic_participant_iou"], s["car_iou"]])
            lines += [
                f"### Decoded-task metrics ({decoded['windows']} windows anchored every "
                f"{decoded['every']}th frame; identical frames for all models at common horizons)",
                "",
                markdown_table(
                    ["horizon / latent source", "AbsRel", "RMSE (m)", "delta1", "vehicle AbsRel",
                     "mIoU", "planning mIoU", "drivable IoU", "drivable bIoU", "traffic IoU", "car IoU"],
                    rows,
                ),
                "",
                "depth scope non-sky unless stated; semantics IoU dataset-level over the sampled frames.",
                "",
            ]
    return "\n".join(lines)


def print_overview(tag: str, result: dict) -> None:
    horizons = result["horizon_seconds"]
    variants = result["latent"]["variants"]
    print(f"\n[{tag}] latent L1 over {result['latent']['windows']} test windows:")
    for k, h in enumerate(horizons):
        print(f"  step +{h:.1f}s | rollout {variants['rollout']['l1'][k]:.4f} | "
              f"one-step (TF) {variants['teacher_forced']['l1'][k]:.4f} | "
              f"copy-last-frame {variants['copy_last']['l1'][k]:.4f} | "
              f"copy-previous {variants['copy_previous']['l1'][k]:.4f} | "
              f"zero-action rollout {variants['rollout_zero_action']['l1'][k]:.4f} | "
              f"rollout/copy {variants['rollout']['l1'][k] / variants['copy_last']['l1'][k]:.3f}")
    print(f"  per sequence (rollout L1 at +{horizons[0]:.1f}s / +{horizons[-1]:.1f}s, copy at the same):")
    for seq, n in result["latent"]["windows_per_sequence"].items():
        r = variants["rollout"]["l1_per_sequence"][seq]
        c = variants["copy_last"]["l1_per_sequence"][seq]
        print(f"    seq {seq} ({n} windows): rollout {r[0]:.4f} / {r[-1]:.4f}   copy {c[0]:.4f} / {c[-1]:.4f}")
    decoded = result.get("decoded")
    if decoded:
        print(f"  decoded-task metrics ({decoded['windows']} windows):")
        for k, h in enumerate(horizons):
            for source in SOURCES:
                d = decoded["depth"][source][k]["non-sky"]
                s = decoded["semantics"][source][k]
                print(f"    +{h:.1f}s {SOURCES[source]:<30} depth AbsRel {d['absrel']:.4f} d1 {d['d1']:.4f} | "
                      f"mIoU {s['miou']:.4f} planning mIoU {s['planning_group_miou']:.4f} "
                      f"drivable IoU {s['drivable_iou']:.4f} traffic IoU {s['traffic_participant_iou']:.4f}")


# ----------------------------------------------------------------------------- main

def evaluate_checkpoint(path: Path, args, depth_decoder, semantic_decoder) -> tuple[str, dict]:
    model, checkpoint = load_world_model(path)
    checkpoint_line = describe_checkpoint(path, checkpoint)
    print(f"\nloaded {checkpoint_line}")

    # The action normalisation is part of the model contract (see train_ac_predictor).
    assert checkpoint["max_speed_mps"] == MAX_SPEED_MPS, "MAX_SPEED_MPS changed since training"
    step_seconds = float(checkpoint["step_seconds"])
    act_scale = action_scale(step_seconds).to(DEVICE)
    frame_stride = frame_stride_for(step_seconds)
    if args.horizon_seconds is not None:
        steps = max(1, int(round(args.horizon_seconds / step_seconds)))
    else:
        steps = args.rollout_steps
    tag = model_tag(checkpoint, path)
    figures_dir = args.figures_dir / tag

    dataset = KITTIRolloutDataset(
        TEST_SEQUENCES,
        context_length=args.context_length,
        future_length=steps,
        frame_stride=frame_stride,
    )
    assert math.isclose(dataset.step_seconds, step_seconds), (dataset.step_seconds, step_seconds)
    print(f"[{tag}] {len(dataset)} test windows: context {args.context_length} x {step_seconds:g}s, "
          f"{steps} rollout steps of {step_seconds:g}s (horizon {steps * step_seconds:.1f}s)")

    latent_set = dataset if args.max_windows is None else Subset(dataset, range(min(args.max_windows, len(dataset))))
    loader = DataLoader(latent_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)
    latent = evaluate_latent(model, loader, act_scale, steps)

    result = {
        "checkpoint": checkpoint_line,
        "checkpoint_path": str(path),
        "iteration": checkpoint.get("iteration"),
        "val_ar": checkpoint.get("val_ar"),
        "step_seconds": step_seconds,
        "frame_stride": frame_stride,
        "context_length": args.context_length,
        "horizon_steps": steps,
        "horizon_seconds": [(k + 1) * step_seconds for k in range(steps)],
        "test_sequences": TEST_SEQUENCES,
        "latent": latent.summary(),
    }
    if not args.skip_decoded:
        print(f"[{tag}] decoding every {args.decoded_every}th window through the frozen decoders ...")
        result["decoded"] = evaluate_decoded(model, dataset, depth_decoder, semantic_decoder, act_scale,
                                             steps, args.decoded_every, args.max_windows)
    print_overview(tag, result)

    figures_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(result, figures_dir / "metrics.json")
    if args.rollouts:
        save_figures(model, dataset, depth_decoder, semantic_decoder, act_scale, steps, args.rollouts, figures_dir)
        print(f"[{tag}] rollout figures saved to {figures_dir}")
    return tag, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained ac-style world models on the test split")
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None,
                        help="world-model checkpoints (default: every checkpoints_wm/world_model_dt*.pt)")
    parser.add_argument("--context-length", type=int, default=4,
                        help="context frames per window (matches the trainer default)")
    parser.add_argument("--rollout-steps", type=int, default=4,
                        help="rollout steps per model (0.2s model -> 0.8s, 0.5s model -> 2.0s)")
    parser.add_argument("--horizon-seconds", type=float, default=None,
                        help="instead of --rollout-steps: roll every model out to this physical horizon")
    parser.add_argument("--rollouts", type=int, default=6,
                        help="evenly-spaced test windows to visualise per model (0 disables)")
    parser.add_argument("--decoded-every", type=int, default=20,
                        help="decode every Nth test window for the decoded-task metrics")
    parser.add_argument("--skip-decoded", action="store_true", help="latent metrics only")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap the windows scored (latent and decoded) -- smoke tests")
    parser.add_argument("--depth-checkpoint", type=Path, default=DEPTH_CHECKPOINT)
    parser.add_argument("--semantics-checkpoint", type=Path, default=SEMANTICS_CHECKPOINT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--replot", action="store_true",
                        help="skip evaluation: rebuild summary.md and the comparison plots from the "
                             "per-model <figures-dir>/<tag>/metrics.json files of an earlier run")
    args = parser.parse_args()

    if args.replot:
        results = {p.parent.name: json.loads(p.read_text())
                   for p in sorted(args.figures_dir.glob("*/metrics.json"))}
        if not results:
            raise SystemExit(f"no <tag>/metrics.json under {args.figures_dir} to replot")
        print("replotting from: " + ", ".join(results))
        write_summary(results, args.figures_dir)
        return

    checkpoints = args.checkpoints or default_checkpoints()
    if not checkpoints:
        raise SystemExit(f"no world-model checkpoints found in {WM_CHECKPOINT_DIR}")

    depth_decoder, depth_ckpt = load_depth_decoder(args.depth_checkpoint)
    semantic_decoder, sem_ckpt = load_semantic_decoder(args.semantics_checkpoint)
    print(f"depth decoder:    {describe_checkpoint(args.depth_checkpoint, depth_ckpt)}")
    print(f"semantic decoder: {describe_checkpoint(args.semantics_checkpoint, sem_ckpt)}")

    results: dict[str, dict] = {}
    for path in checkpoints:
        tag, result = evaluate_checkpoint(path, args, depth_decoder, semantic_decoder)
        if tag in results:
            raise SystemExit(f"duplicate model tag {tag!r} for {path}: results would overwrite each other")
        results[tag] = result

    write_summary(results, args.figures_dir)


def write_summary(results: dict[str, dict], figures_dir: Path) -> None:
    """Cross-model outputs: comparison plots, summary.md and the combined metrics.json."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_latent_curves(results, figures_dir / "latent_l1_vs_horizon.png")
    if any("decoded" in r for r in results.values()):
        plot_decoded_curves(results, figures_dir / "decoded_metrics_vs_horizon.png")
    (figures_dir / "summary.md").write_text(summary_markdown(results))
    write_metrics_json({"models": results}, figures_dir / "metrics.json")
    print(f"\nsummary + comparison plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
