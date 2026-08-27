"""
Evaluate the trained geometry-guided ProjectedPredictor world models on the
held-out KITTI test sequences (SPLIT_V1.test_sequences, see data/splits.py)
with the protocol of ac_predictor_evaluator.py -- same windows, same latent
variants, same decoded-task metrics on the same anchor frames, same figures
and output layout -- so the two architectures compare number for number.

    PYTHONPATH=src python -m jepa_drive_wm.evals.projected_predictor_evaluator \
        [--checkpoints outputs/checkpoints_projected_predictor/projected_predictor_dt0.2s.pt ...]

By default every `projected_predictor_dt*.pt` in
OUTPUTS_DIR/checkpoints_projected_predictor is evaluated. Each checkpoint
records its step duration (`step_seconds`), so the frame stride of the test
windows is derived per model (0.2 s -> stride 2, 0.5 s -> stride 5). The
architecture fixes the window shape: five context frames I_{t-4..t} and
exactly four predicted steps (ProjectedPredictor.NUM_FUTURE_STEPS) -- the ac
evaluator's default of 4 rollout steps (0.8 s for the 0.2 s model, 2.0 s for
the 0.5 s model); --rollout-steps can only shorten that. Curves are plotted
against horizon in seconds, as there.

How the ac protocol maps onto this architecture:

- inputs are raw RGB, the FoundationStereo depth of I_t and source-relative
  ego motions (KITTIProjectedPredictorDataset), not cached latents. The model
  encodes I_t and the true future frames itself with the frozen V-JEPA 2.1
  ViT-B in image mode -- the same encoder at the same 384 x 1248 resolution
  that produced the cached vjepa_vitb latents the ac models and the frozen
  decoders were trained on. Targets and the copy baselines are the model's
  own encodings (what it was trained against); the first window of every run
  reports their agreement with the cache (`target_vs_cache` in metrics.json).
  The two differ only by the image resize (PIL antialiased bilinear in the
  cache builder vs F.interpolate bilinear in the dataset): measured 2026-08-22
  on test windows, mean |z_model - z_cache| ~ 0.02 on latents of mean |z|
  ~ 0.53, cosine 0.9995, and the copy-last L1 agrees to < 0.001 whether
  computed in model or cache space -- the latent numbers here and in
  evals_wm are on the same scale;
- the prediction is Z_hat = Z0 + DeltaZ: a deterministic warp+copy proposal
  plus a learned correction. The proposal is this architecture's own
  initialisation prior (zero-init head, so Z_hat == Z0 before training) --
  what `copy_last` is for the ac models -- and is reported as an extra
  latent variant and decoded source, `proposal`, next to the copy baselines
  shared with the ac evaluator;
- `teacher_forced`: the same warps and corrections, but the proposal's
  copy-forward infill of geometrically missing patches takes the *true*
  previous latent instead of the previous prediction. That infill is the only
  place the model's own history enters (the warps and the predictor's hidden
  states come from the observed frames alone), so this is the exact
  one-step-from-true-history analogue and, as for the ac models, step 0
  coincides with the rollout;
- `rollout_zero_action`: the model re-run with all four ego motions zeroed
  (identity warp);
- geometry splits every future frame into patches the warp reaches (valid)
  and patches it cannot (missing, infilled by copy-forward). The latent L1 is
  additionally reported per region, pooled over patches -- the trainer's
  context / masked decomposition -- with the valid-patch fraction per step.

Per model, over every test window (or every --latent-every-th: each window
costs two model forwards, true and zeroed motion, ~3 s on the 4070 laptop --
the PyTorch3D warps are half of it -- so the full ~3.7k-window pass takes
~3 h per model, against minutes for the ac evaluator; --latent-every 2 or 3
halves/thirds that at little statistical cost since the windows overlap):

- latent-space metrics: mean L1 in the frozen V-JEPA 2.1 space per step for
      rollout              autoregressive (step k infills from prediction k-1)
      teacher_forced       one-step from the true history (see above)
      proposal             warp+copy Z0 alone, no learned correction
      copy_last            copy the last context frame
      copy_previous        copy the true frame one step back (the matching
                           baseline for teacher_forced)
      rollout_zero_action  rollout with the ego motion zeroed
  overall, per test sequence (01 highway / 07 urban / 09 rural) and per
  geometric region;
- decoded-task metrics on the shared anchor-frame grid (--decoded-every; the
  same frames the ac models are scored on at common horizons): the true,
  copy-last, proposal and predicted latents pushed through the frozen depth
  and semantic decoders and scored against the pseudolabels with the decoder
  evaluators' accumulators (DepthMetricAccumulator / SemanticMetricAccumulator);
- figures: for a few windows, the rollout decoded to depth and semantics: a
  first column for the last context frame t, then one column per step, with
  rows camera, RGB warp of I_t (the geometric prior; missing patches black),
  pseudolabel GT, decoder on the true / copy-last / proposal / predicted
  latent.

Outputs in OUTPUTS_DIR/evals_projected_predictor (the evals_wm layout):
    <tag>/metrics.json                 everything for one model (tag = dt0.2s, dt0.5s for the
                                       trainer's default filenames; other files get the
                                       checkpoint stem appended so same-step runs never collide)
    <tag>/seqXX_windowNNNNNN_{depth,semantics}.png
    latent_l1_vs_horizon.png           all models: rollout / TF / proposal / copy / zero-action L1
    latent_l1_by_region.png            all models: L1 on warp-valid vs missing patches vs horizon
    decoded_metrics_vs_horizon.png     all models: depth AbsRel, delta1, mIoU, planning mIoU
    summary.md                         tables for all models
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from jepa_drive_wm.data.data_interface_projected_predictor import KITTIProjectedPredictorDataset
from jepa_drive_wm.data.splits import SPLIT_V1
# Shared with the ac evaluator by import, not by copy: the anchor-frame grid,
# the figure-window picks, the decode step and the figure grid are what make
# the two evaluators score and show identical frames.
from jepa_drive_wm.evals.ac_predictor_evaluator import (
    FIGURE_ANCHOR_GRID,
    MAX_FIGURE_COLUMNS,
    SOURCE_STYLE as AC_SOURCE_STYLE,
    VARIANT_STYLE as AC_VARIANT_STYLE,
    _anchor_items,
    _decode,
    _future_frame,
    _save_grid,
    frame_stride_for,
)
from jepa_drive_wm.evals.common import (
    DEPTH_CHECKPOINT,
    DEVICE,
    MUTED,
    PROJECTED_CHECKPOINT_DIR,
    SEMANTICS_CHECKPOINT,
    SERIES,
    describe_checkpoint,
    finish_figure,
    load_depth_decoder,
    load_semantic_decoder,
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
from jepa_drive_wm.models.predictors.projected_predictor.projected_predictor import (
    ProjectedPredictor,
    ProjectedPredictorOutput,
)
from jepa_drive_wm.models.predictors.projected_predictor.warp_rgb import (
    DEFAULT_PATCH_COVERAGE_THRESHOLD,
    DEFAULT_RADIUS_PX,
)
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.train.train_depth import MAX_DEPTH, MIN_DEPTH
from jepa_drive_wm.train.train_projected_predictor import (
    DEFAULT_VJEPA_CHECKPOINT,
    IMAGE_SIZE,
    PATCH_SIZE,
    build_model,
)
from jepa_drive_wm.training_utils import autocast
from jepa_drive_wm.viz.visualiser import class_colors

FIGURES_DIR = OUTPUTS_DIR / "evals_projected_predictor"
TEST_SEQUENCES = list(SPLIT_V1.test_sequences)
# The architecture fixes the window: 5 context frames, 4 predicted steps on the 24 x 78 token grid.
CONTEXT_LENGTH = 5
MAX_STEPS = ProjectedPredictor.NUM_FUTURE_STEPS
GRID_H, GRID_W = IMAGE_SIZE[0] // PATCH_SIZE, IMAGE_SIZE[1] // PATCH_SIZE

# Latent-space variants, in display order: the ac evaluator's five plus the proposal.
VARIANTS = {
    "rollout": "autoregressive rollout",
    "teacher_forced": "one-step (teacher forced)",
    "proposal": "warp+copy proposal (no correction)",
    "copy_last": "copy last context frame",
    "copy_previous": "copy previous true frame",
    "rollout_zero_action": "rollout, zero action",
}
# Latent sources fed to the frozen decoders, in display order.
SOURCES = {
    "true": "true latent (decoder ceiling)",
    "copy": "copy-last-frame latent",
    "proposal": "warp+copy proposal latent",
    "predicted": "predicted latent (rollout)",
}
# Geometric regions of a future frame, from the true-motion warp of I_t.
REGIONS = {"valid": "warp-valid patches", "missing": "geometrically missing patches"}


def default_checkpoints() -> list[Path]:
    """Every horizon-tagged projected-predictor checkpoint the trainer can have written."""
    return sorted(PROJECTED_CHECKPOINT_DIR.glob("projected_predictor_dt*.pt"))


def model_tag(checkpoint: dict, path: Path) -> str:
    """
    Names the run, its results entry and its output folder. The trainer's
    default filename projected_predictor_<step tag>.pt gets the bare step tag
    (dt0.2s, dt0.5s); any other file keeps its stem too, so two runs at the
    same step never overwrite each other.
    """
    tag = f"dt{checkpoint['step_seconds']:g}s"
    return tag if path.stem == f"projected_predictor_{tag}" else f"{tag}_{path.stem}"


# ----------------------------------------------------------------------------- model

def load_projected_predictor(
    path: Path,
    vjepa_checkpoint: Path | None = None,
    radius_px: float = DEFAULT_RADIUS_PX,
    patch_coverage_threshold: float = DEFAULT_PATCH_COVERAGE_THRESHOLD,
) -> tuple[ProjectedPredictor, dict]:
    """
    Rebuild a trained ProjectedPredictor: the frozen released ViT-B encoder and
    predictor from the V-JEPA checkpoint (exactly as the trainer's build_model),
    then the post-trained predictor body and the correction head from ours.

    The checkpoint stores the V-JEPA checkpoint path it was trained from; when
    that path does not exist on this machine (trained elsewhere) the repo
    default is used. The warp geometry (splat radius, patch coverage
    threshold) is not recorded in the checkpoint, so it has to be the
    trainer's -- the defaults unless training was run with overrides.
    """
    # weights_only=False: plain dict metadata (val metrics, paths) next to the state dicts.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.pop("optimizer", None)  # not needed for evaluation; frees host memory
    if vjepa_checkpoint is None:
        recorded = Path(checkpoint.get("base_vjepa_checkpoint", DEFAULT_VJEPA_CHECKPOINT))
        vjepa_checkpoint = recorded if recorded.exists() else DEFAULT_VJEPA_CHECKPOINT
    model = build_model(
        device=DEVICE,
        vjepa_checkpoint=vjepa_checkpoint,
        radius_px=radius_px,
        patch_coverage_threshold=patch_coverage_threshold,
        finetune_predictor=False,
    )
    model.predictor.load_state_dict(checkpoint["predictor"])
    model.correction_head.load_state_dict(checkpoint["correction_head"])
    model.eval()
    checkpoint["vjepa_checkpoint_used"] = str(vjepa_checkpoint)
    return model, checkpoint


def unbatch(batch: dict) -> dict:
    """A batch-size-one DataLoader batch -> the plain dataset sample (the model is per-sample)."""
    return {key: value[0] for key, value in batch.items()}


@torch.no_grad()
def run_model(
    model: ProjectedPredictor,
    sample: dict,
    *,
    ego_motions: torch.Tensor | None = None,
    encode_targets: bool = True,
) -> ProjectedPredictorOutput:
    """
    One window (an unbatched dataset sample) through the model under the
    trainer's autocast, after installing the window's camera intrinsics in the
    warper (train_projected_predictor._set_batch_intrinsics does the same per
    batch). `ego_motions` replaces the window's source-relative motions (the
    zero-action variant); encode_targets=False skips re-encoding the true
    future frames when the targets are already known.
    """
    model.warp_module.K.copy_(sample["intrinsics"].to(model.warp_module.K))
    with autocast(DEVICE):
        return model(
            context_rgb=sample["context_rgb"],
            depth_t=sample["depth_t"],
            ego_motions=sample["ego_motions"] if ego_motions is None else ego_motions,
            future_rgb=sample["future_rgb"] if encode_targets else None,
        )


def teacher_forced_predictions(
    model: ProjectedPredictor, output: ProjectedPredictorOutput, target: torch.Tensor
) -> torch.Tensor:
    """
    One-step predictions from the true history, [1, K, N, C] fp32: the
    rollout's own warps and corrections, but the proposal of step k infills its
    geometrically missing patches from the true latent of step k-1 instead of
    prediction k-1 (step 0 infills from z_t, exactly as the rollout does).
    """
    warped, valid = output.warped_latents.float(), output.patch_valid
    corrections, z_t = output.corrections.float(), output.z_t.float()
    predictions = []
    for k in range(target.shape[1]):
        z_previous = z_t if k == 0 else target[:, k - 1]
        z0 = model.latent_proposal(z_previous=z_previous, z_warped=warped[:, k], patch_valid=valid[:, k])
        predictions.append(z0 + corrections[:, k])
    return torch.stack(predictions, dim=1)


def latent_sources(output: ProjectedPredictorOutput) -> dict[str, torch.Tensor]:
    """The latents the frozen decoders see at every step, each [K, N, C] fp32 (on DEVICE)."""
    steps = output.predictions.shape[1]
    return {
        "true": output.target_latents.float()[0],
        "copy": output.z_t.float().expand(steps, -1, -1),
        "proposal": output.proposals.float()[0],
        "predicted": output.predictions.float()[0],
    }


def to_chw(latent_nc: torch.Tensor) -> torch.Tensor:
    """[N, C] row-major tokens -> [1, C, H, W] grid, the decoders' input."""
    return latent_nc.reshape(GRID_H, GRID_W, -1).permute(2, 0, 1)[None]


def image(frame: torch.Tensor) -> np.ndarray:
    """(3, H, W) or (H, W, 3) RGB tensor in [0, 1] -> displayable numpy image."""
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.permute(1, 2, 0)
    return frame.detach().float().clamp(0, 1).cpu().numpy()


def cache_agreement(dataset: KITTIProjectedPredictorDataset, sample: dict,
                    output: ProjectedPredictorOutput) -> dict[str, float] | None:
    """
    How far the model's encodings of this window's true future frames are from
    the cached vjepa_vitb latents: the model is scored in its own encoder's
    space, the decoders (and the ac models) use the cache, and the two are
    meant to be the same encoder at the same resolution. Mean |difference|,
    the mean |latent| for scale and the per-token cosine similarity. None when
    the cache is not available on this machine.
    """
    sequence = dataset.sequences[int(sample["sequence_nr"])]
    try:
        cached = torch.stack([torch.from_numpy(sequence.get_vjepa_features(int(frame)))
                              for frame in sample["future_frame_ids"]])
    except FileNotFoundError:
        return None
    encoded = output.target_latents.float()[0].cpu()
    return {
        "l1": float((encoded - cached).abs().mean()),
        "target_abs_mean": float(encoded.abs().mean()),
        "cosine": float(torch.nn.functional.cosine_similarity(encoded, cached, dim=-1).mean()),
    }


# ----------------------------------------------------------------------------- latent metrics

class LatentMetrics:
    """
    Per-step mean latent L1 for each variant, pooled over windows and also per
    sequence (the ac evaluator's quantity), plus -- new here -- the same L1
    pooled over patches within each geometric region of a step (warp-valid /
    missing, defined by the true-motion warp) and the valid-patch fraction.
    """

    def __init__(self, steps: int, variants: list[str]) -> None:
        self.steps = steps
        self.variants = list(variants)
        self._sum: dict[str, np.ndarray] = {}
        self._seq_sum: dict[str, dict[int, np.ndarray]] = {}
        self._windows = 0
        self._seq_windows: dict[int, int] = {}
        self._region_sum: dict[str, dict[str, np.ndarray]] = {}
        self._region_count = {region: np.zeros(steps) for region in REGIONS}

    def update(self, variant: str, patch_l1: torch.Tensor, patch_valid: torch.Tensor,
               sequence_nrs: torch.Tensor) -> None:
        """
        patch_l1:    [B, K, N] mean |pred - target| over channels per patch (fp32, cpu)
        patch_valid: [B, K, N] bool, the warp validity of each step's patches (cpu)
        """
        values = patch_l1.double()
        # mean over (H, W, C) per window and step, like latent_loss but kept per window
        per_window = values.mean(dim=-1).numpy()
        self._sum[variant] = self._sum.get(variant, np.zeros(self.steps)) + per_window.sum(axis=0)
        seq_sum = self._seq_sum.setdefault(variant, {})
        for row, seq in zip(per_window, sequence_nrs.tolist()):
            seq_sum[seq] = seq_sum.get(seq, np.zeros(self.steps)) + row

        masks = {"valid": patch_valid.bool(), "missing": ~patch_valid.bool()}
        region_sum = self._region_sum.setdefault(variant, {region: np.zeros(self.steps) for region in REGIONS})
        for region, mask in masks.items():
            region_sum[region] += (values * mask).sum(dim=(0, 2)).numpy()

        if variant == "rollout":  # count windows and patches once, not once per variant
            self._windows += per_window.shape[0]
            for seq in sequence_nrs.tolist():
                self._seq_windows[seq] = self._seq_windows.get(seq, 0) + 1
            for region, mask in masks.items():
                self._region_count[region] += mask.sum(dim=(0, 2)).double().numpy()

    @property
    def windows(self) -> int:
        return self._windows

    def mean(self, variant: str) -> np.ndarray:
        return self._sum[variant] / max(self._windows, 1)

    def per_sequence(self, variant: str) -> dict[int, np.ndarray]:
        return {seq: self._seq_sum[variant][seq] / self._seq_windows[seq]
                for seq in sorted(self._seq_windows)}

    def region_mean(self, variant: str, region: str) -> np.ndarray:
        """L1 pooled over all patches of that region (not a mean of per-window means)."""
        return self._region_sum[variant][region] / np.maximum(self._region_count[region], 1)

    def valid_fraction(self) -> np.ndarray:
        total = self._region_count["valid"] + self._region_count["missing"]
        return self._region_count["valid"] / np.maximum(total, 1)

    def summary(self) -> dict:
        variants = [v for v in self.variants if v in self._sum]
        return {
            "windows": self._windows,
            "windows_per_sequence": {f"{s:02d}": n for s, n in sorted(self._seq_windows.items())},
            "variants": {
                variant: {
                    "l1": self.mean(variant).tolist(),
                    "l1_mean_over_steps": float(self.mean(variant).mean()),
                    "l1_per_sequence": {f"{s:02d}": v.tolist() for s, v in self.per_sequence(variant).items()},
                }
                for variant in variants
            },
            "regions": {
                "definition": "valid = patches the true-motion RGB warp of I_t covers (coverage >= threshold), "
                              "missing = the rest (copy-forward infilled); L1 pooled over patches",
                "valid_patch_fraction": self.valid_fraction().tolist(),
                "l1": {variant: {region: self.region_mean(variant, region).tolist() for region in REGIONS}
                       for variant in variants},
            },
        }


@torch.no_grad()
def evaluate_latent(model: ProjectedPredictor, loader, steps: int) -> tuple[LatentMetrics, dict | None]:
    """
    One pass over the test windows: per-step L1 for every variant in VARIANTS
    (two model forwards per window: the true motions and the zeroed ones).
    Also returns the first window's target-vs-cache agreement.
    """
    model.eval()
    metrics = LatentMetrics(steps, list(VARIANTS))
    dataset = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
    agreement = None
    start = time.time()

    for n, batch in enumerate(loader):
        sample = unbatch(batch)
        output = run_model(model, sample)
        zero = run_model(model, sample, ego_motions=torch.zeros_like(sample["ego_motions"]),
                         encode_targets=False)
        if n == 0:
            agreement = cache_agreement(dataset, sample, output)

        target = output.target_latents.float()[:, :steps]            # [1, K, N, C]
        valid = output.patch_valid[:, :steps].cpu()                   # [1, K, N]
        z_t = output.z_t.float()[:, None]                             # [1, 1, N, C]
        predictions = {
            "rollout": output.predictions.float()[:, :steps],
            "teacher_forced": teacher_forced_predictions(model, output, target),
            "proposal": output.proposals.float()[:, :steps],
            "copy_last": z_t.expand_as(target),
            # one-step baseline: the true frame one step back (z_t for step 0)
            "copy_previous": torch.cat([z_t, target[:, :-1]], dim=1),
            "rollout_zero_action": zero.predictions.float()[:, :steps],
        }
        for variant, pred in predictions.items():
            patch_l1 = (pred - target).abs().mean(dim=-1).cpu()      # [1, K, N], fp32
            metrics.update(variant, patch_l1, valid, batch["sequence_nr"])

        if (n + 1) % 100 == 0 or n + 1 == len(loader):
            rate = (n + 1) / (time.time() - start)
            print(f"  latent {n + 1}/{len(loader)} windows ({rate:.2f} windows/s, "
                  f"rollout L1 so far {metrics.mean('rollout').round(4).tolist()})", flush=True)
    return metrics, agreement


# ----------------------------------------------------------------------------- decoded metrics

@torch.no_grad()
def evaluate_decoded(
    model: ProjectedPredictor,
    dataset: KITTIProjectedPredictorDataset,
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
    steps: int,
    every: int,
    num_workers: int,
    max_windows: int | None = None,
) -> dict:
    """
    Windows anchored on every `every`-th frame of each test sequence (the ac
    evaluator's _anchor_items, so the very same frames): run the model, decode
    the true / copy / proposal / predicted latents at each step with the frozen
    decoders and score them against the pseudolabels of that step's frame.
    """
    model.eval()
    depth_acc = {source: [DepthMetricAccumulator() for _ in range(steps)] for source in SOURCES}
    sem_acc = {source: [SemanticMetricAccumulator() for _ in range(steps)] for source in SOURCES}

    items = _anchor_items(dataset, every)
    if max_windows is not None:
        items = items[:max_windows]
    loader = DataLoader(Subset(dataset, items), batch_size=1, shuffle=False, num_workers=num_workers)

    for n, batch in enumerate(loader):
        sample = unbatch(batch)
        sequence = dataset.sequences[int(sample["sequence_nr"])]
        start_index = int(sample["start_index"])
        latents = latent_sources(run_model(model, sample))

        for k in range(steps):
            frame = _future_frame(dataset, start_index, k)
            gt_depth = sequence.get_depth(frame)
            gt_semantics = sequence.get_semantics(frame)
            for source, latent in latents.items():
                depth, semantics = _decode(depth_decoder, semantic_decoder, to_chw(latent[k]),
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

@torch.no_grad()
def save_figures(
    model: ProjectedPredictor,
    dataset: KITTIProjectedPredictorDataset,
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
    steps: int,
    n_rollouts: int,
    figures_dir: Path,
) -> None:
    """
    For n_rollouts test windows evenly spaced over the FIGURE_ANCHOR_GRID
    anchors (the ac evaluator's picks, so both architectures show the same
    scenes): one depth figure and one semantics figure. Columns: the last
    context frame t (camera I_t, I_t itself in the warp row, pseudolabel GT at
    t, the decoders on z_t -- identical in the four decoder rows, since copy,
    proposal and rollout all start from z_t), then up to MAX_FIGURE_COLUMNS
    rollout steps. Rows: camera, RGB warp of I_t (missing patches black),
    pseudolabel GT, decoder on the true / copy-last / proposal / predicted
    latent.
    """
    shown = sorted(set(np.linspace(0, steps - 1, min(steps, MAX_FIGURE_COLUMNS)).round().astype(int)))
    column_titles = ["t (last context)"] + [f"t + {(k + 1) * dataset.step_seconds:.1f}s" for k in shown]
    candidates = _anchor_items(dataset, FIGURE_ANCHOR_GRID)
    picks = sorted(set(np.linspace(0, len(candidates) - 1, min(n_rollouts, len(candidates))).round().astype(int)))

    for item in (candidates[i] for i in picks):
        sample = dataset[item]
        sequence_nr = int(sample["sequence_nr"])
        start_index = int(sample["start_index"])
        sequence = dataset.sequences[sequence_nr]

        output = run_model(model, sample)
        latents = latent_sources(output)
        copy_chw = to_chw(latents["copy"][0])

        # column 0: the last context frame t, everything decoded from z_t
        anchor = _future_frame(dataset, start_index, -1)
        gt_depth_t = sequence.get_depth(anchor)
        gt_semantics_t = sequence.get_semantics(anchor)
        depth_t, semantics_t = _decode(depth_decoder, semantic_decoder, copy_chw,
                                       gt_depth_t.shape, gt_semantics_t.shape)
        cameras, warps = [sequence.get_image(anchor)], [image(sample["context_rgb"][-1])]
        depth_gt, sem_gt = [gt_depth_t], [class_colors(gt_semantics_t)]
        depth_by = {source: [depth_t] for source in SOURCES}
        sem_by = {source: [class_colors(semantics_t)] for source in SOURCES}
        for k in shown:
            frame = _future_frame(dataset, start_index, k)
            gt_depth = sequence.get_depth(frame)
            gt_semantics = sequence.get_semantics(frame)
            cameras.append(sequence.get_image(frame))
            warps.append(image(output.streams.warps[k].rgb_patch_masked))
            depth_gt.append(gt_depth)
            sem_gt.append(class_colors(gt_semantics))
            for source, latent in latents.items():
                depth, semantics = _decode(depth_decoder, semantic_decoder, to_chw(latent[k]),
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
                ("RGB warp of I_t\n(missing patches black)", warps, {}),
                ("FoundationStereo\ndepth", depth_gt, depth_kwargs),
                ("depth from\ntrue latent", depth_by["true"], depth_kwargs),
                ("depth from\ncopy-last latent", depth_by["copy"], depth_kwargs),
                ("depth from\nproposal latent", depth_by["proposal"], depth_kwargs),
                ("depth from\npredicted latent", depth_by["predicted"], depth_kwargs),
            ],
            column_titles, title, figures_dir / f"{stem}_depth.png",
        )
        _save_grid(
            [
                ("camera", cameras, {}),
                ("RGB warp of I_t\n(missing patches black)", warps, {}),
                ("OneFormer\nsemantics", sem_gt, {}),
                ("semantics from\ntrue latent", sem_by["true"], {}),
                ("semantics from\ncopy-last latent", sem_by["copy"], {}),
                ("semantics from\nproposal latent", sem_by["proposal"], {}),
                ("semantics from\npredicted latent", sem_by["predicted"], {}),
            ],
            column_titles, title, figures_dir / f"{stem}_semantics.png",
        )


# ----------------------------------------------------------------------------- comparison plots

# The ac evaluator's styles for the shared variants / sources, plus the proposal.
VARIANT_STYLE = {**AC_VARIANT_STYLE,
                 "proposal": dict(linestyle=(0, (4, 1.5, 1, 1.5)), linewidth=1.6, marker="D", markersize=3.5)}
SOURCE_STYLE = {**AC_SOURCE_STYLE,
                "proposal": dict(linestyle=(0, (4, 1.5, 1, 1.5)), linewidth=1.6, marker="D", markersize=3.5)}
# Which copy baseline each prediction is judged against (same context -> same baseline).
BASELINE_OF = {"rollout": "copy_last", "rollout_zero_action": "copy_last", "proposal": "copy_last",
               "teacher_forced": "copy_previous"}
# Variants shown in the per-region plot / table: the prediction, its own prior, the ac prior.
REGION_VARIANTS = ("rollout", "proposal", "copy_last")


def plot_latent_curves(results: dict[str, dict], path: Path) -> None:
    """
    Left: latent L1 vs horizon for every model (colour) and variant (line style).
    Right: each prediction relative to its matching copy baseline (1.0 = no
    better than copying; below = learned dynamics): rollouts, zero-action
    rollouts and the proposal against copy-last (same context), teacher
    forcing against copy-previous.
    """
    fig, axes = new_figure(ncols=2, width=6.4, height=4.4)
    ax_abs, ax_rel = axes[0]
    for i, (tag, result) in enumerate(results.items()):
        color = SERIES[i % len(SERIES)]
        horizons = np.asarray(result["horizon_seconds"])
        variants = result["latent"]["variants"]
        for variant in VARIANTS:
            if variant not in variants:
                continue
            style = VARIANT_STYLE[variant]
            values = np.asarray(variants[variant]["l1"])
            ax_abs.plot(horizons, values, color=color, label=f"{tag} - {VARIANTS[variant]}", **style)
            if variant in BASELINE_OF and BASELINE_OF[variant] in variants:
                baseline = np.asarray(variants[BASELINE_OF[variant]]["l1"])
                ax_rel.plot(horizons, values / baseline, color=color,
                            label=f"{tag} - {VARIANTS[variant]}", **style)
    ax_rel.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=":")
    ax_rel.text(ax_rel.get_xlim()[1], 1.0, " copy baseline", va="center", ha="right", fontsize=8, color=MUTED)
    ax_rel.text(0.02, 0.95, "rollouts, proposal / copy-last;  one-step / copy-previous",
                transform=ax_rel.transAxes, fontsize=8, color=MUTED, ha="left", va="top")
    style_axes(ax_abs, title="Latent L1 vs horizon (test set)", xlabel="horizon (s)",
               ylabel="mean |pred - true| in V-JEPA space")
    style_axes(ax_rel, title="Relative to the matching copy baseline", xlabel="horizon (s)",
               ylabel="L1 / copy-baseline L1")
    # colour = model, line style = variant; one legend for both panels
    bottom = shared_legend_below(fig, ax_abs, ncol=4)
    finish_figure(fig, path, rect=(0, bottom, 1, 1))


def plot_region_curves(results: dict[str, dict], path: Path) -> None:
    """
    Latent L1 pooled over patches within each geometric region vs horizon:
    left the warp-valid patches (where the proposal is the warped latent),
    right the missing ones (where it is copy-forward) -- rollout, proposal and
    copy-last per model -- with the valid-patch fraction as a dotted curve on
    the right axis of the left panel.
    """
    fig, axes = new_figure(ncols=2, width=6.4, height=4.4)
    ax_frac = axes[0, 0].twinx()
    for i, (tag, result) in enumerate(results.items()):
        color = SERIES[i % len(SERIES)]
        horizons = np.asarray(result["horizon_seconds"])
        regions = result["latent"]["regions"]
        for ax, region in zip(axes[0], REGIONS):
            for variant in REGION_VARIANTS:
                if variant in regions["l1"]:
                    ax.plot(horizons, regions["l1"][variant][region], color=color,
                            label=f"{tag} - {VARIANTS[variant]}", **VARIANT_STYLE[variant])
        ax_frac.plot(horizons, 100 * np.asarray(regions["valid_patch_fraction"]), color=color,
                     linestyle=":", linewidth=1.0, marker=".", markersize=3)
    for ax, region in zip(axes[0], REGIONS):
        style_axes(ax, title=f"Latent L1 on {REGIONS[region]}", xlabel="horizon (s)",
                   ylabel="mean |pred - true| in V-JEPA space (pooled over patches)")
    ax_frac.set_ylabel("valid patches (%)  [dotted]", color=MUTED, fontsize=9)
    ax_frac.tick_params(colors=MUTED, labelsize=8)
    ax_frac.set_ylim(0, 100)
    ax_frac.spines["top"].set_visible(False)
    bottom = shared_legend_below(fig, axes[0, 0], ncol=3)
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
            for source in SOURCES:
                per_step = decoded[task][source]
                values = [(m[scope][key] if scope else m[key]) for m in per_step]
                ax.plot(horizons, values, color=color, label=f"{tag} - {SOURCES[source]}", **SOURCE_STYLE[source])
        style_axes(ax, title=title, xlabel="horizon (s)", ylabel=key)
    # colour = model, line style = latent source; one legend for all four panels
    bottom = shared_legend_below(fig, axes[0, 0], ncol=4)
    finish_figure(fig, path, rect=(0, bottom, 1, 1))


# ----------------------------------------------------------------------------- reporting

def summary_markdown(results: dict[str, dict]) -> str:
    lines = [
        "# projected predictor world models -- test set (SPLIT_V1: sequences "
        + ", ".join(f"{s:02d}" for s in TEST_SEQUENCES) + ")",
        "",
        "Latent L1 = mean |pred - true| in the frozen V-JEPA 2.1 space, pooled over all test windows "
        "(targets = the model's own frozen ViT-B image-mode encodings of the true future frames, as in "
        "training). `proposal` is the deterministic warp+copy Z0 without the learned correction -- the "
        "model's own initialisation prior; `teacher_forced` infills the proposal's missing patches from "
        "the true previous latent instead of the previous prediction. Decoded metrics push latents "
        "through the frozen depth / semantic decoders and score them against the pseudolabels with the "
        "decoder evaluators' code, on the same anchor frames as the ac evaluator.",
        "",
    ]
    for tag, result in results.items():
        horizons = result["horizon_seconds"]
        variants = result["latent"]["variants"]
        regions = result["latent"]["regions"]
        agreement = result.get("target_vs_cache")
        lines += [
            f"## {tag}",
            "",
            f"checkpoint: `{result['checkpoint']}`  ",
            f"context {result['context_length']} frames, frame stride {result['frame_stride']}, "
            f"{result['horizon_steps']} rollout steps of {result['step_seconds']:g}s, "
            f"{result['latent']['windows']} test windows"
            + (f" (every {result['latent_every']}th)" if result.get("latent_every", 1) != 1 else "")
            + f"; warp radius {result['geometry']['radius_px']:g}px, patch coverage threshold "
            f"{result['geometry']['patch_coverage_threshold']:g}  ",
            "model encodings vs cached vjepa_vitb latents (first window): "
            + (f"mean |diff| {agreement['l1']:.4f} on mean |z| {agreement['target_abs_mean']:.3f}, "
               f"cosine {agreement['cosine']:.4f}" if agreement else "n/a (cache not found)"),
            "",
            "### Latent L1 per step",
            "",
            markdown_table(
                ["horizon (s)", *[VARIANTS[v] for v in VARIANTS if v in variants],
                 "rollout / copy", "rollout / proposal"],
                [[f"+{h:.1f}", *[variants[v]["l1"][k] for v in VARIANTS if v in variants],
                  variants["rollout"]["l1"][k] / variants["copy_last"]["l1"][k],
                  variants["rollout"]["l1"][k] / variants["proposal"]["l1"][k]]
                 for k, h in enumerate(horizons)],
            ),
            "",
            "### Latent L1 by geometric region (pooled over patches)",
            "",
            markdown_table(
                ["horizon (s)", "valid patches",
                 *[f"{VARIANTS[v]}, {r}" for r in REGIONS for v in REGION_VARIANTS if v in regions["l1"]]],
                [[f"+{h:.1f}", f"{100 * regions['valid_patch_fraction'][k]:.1f}%",
                  *[regions["l1"][v][r][k] for r in REGIONS for v in REGION_VARIANTS if v in regions["l1"]]]
                 for k, h in enumerate(horizons)],
            ),
            "",
            "### Rollout L1 per sequence",
            "",
            markdown_table(
                ["seq", "windows", *[f"rollout +{h:.1f}s" for h in horizons],
                 *[f"proposal +{h:.1f}s" for h in (horizons[0], horizons[-1])],
                 *[f"copy +{h:.1f}s" for h in (horizons[0], horizons[-1])]],
                [[seq, result["latent"]["windows_per_sequence"][seq],
                  *variants["rollout"]["l1_per_sequence"][seq],
                  variants["proposal"]["l1_per_sequence"][seq][0],
                  variants["proposal"]["l1_per_sequence"][seq][-1],
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
    regions = result["latent"]["regions"]
    print(f"\n[{tag}] latent L1 over {result['latent']['windows']} test windows:")
    for k, h in enumerate(horizons):
        print(f"  step +{h:.1f}s | rollout {variants['rollout']['l1'][k]:.4f} | "
              f"one-step (TF) {variants['teacher_forced']['l1'][k]:.4f} | "
              f"proposal {variants['proposal']['l1'][k]:.4f} | "
              f"copy-last-frame {variants['copy_last']['l1'][k]:.4f} | "
              f"copy-previous {variants['copy_previous']['l1'][k]:.4f} | "
              f"zero-action rollout {variants['rollout_zero_action']['l1'][k]:.4f} | "
              f"rollout/copy {variants['rollout']['l1'][k] / variants['copy_last']['l1'][k]:.3f} | "
              f"rollout/proposal {variants['rollout']['l1'][k] / variants['proposal']['l1'][k]:.3f}")
    print("  by region (L1 pooled over patches; valid = warp-covered, missing = copy-forward infilled):")
    for k, h in enumerate(horizons):
        cells = [f"{VARIANTS[v]} {regions['l1'][v][r][k]:.4f}" for r in REGIONS for v in REGION_VARIANTS]
        print(f"    +{h:.1f}s valid {100 * regions['valid_patch_fraction'][k]:4.1f}% | "
              f"valid: {', '.join(cells[:len(REGION_VARIANTS)])} | "
              f"missing: {', '.join(cells[len(REGION_VARIANTS):])}")
    print(f"  per sequence (rollout L1 at +{horizons[0]:.1f}s / +{horizons[-1]:.1f}s, proposal and copy at the same):")
    for seq, n in result["latent"]["windows_per_sequence"].items():
        r = variants["rollout"]["l1_per_sequence"][seq]
        p = variants["proposal"]["l1_per_sequence"][seq]
        c = variants["copy_last"]["l1_per_sequence"][seq]
        print(f"    seq {seq} ({n} windows): rollout {r[0]:.4f} / {r[-1]:.4f}   "
              f"proposal {p[0]:.4f} / {p[-1]:.4f}   copy {c[0]:.4f} / {c[-1]:.4f}")
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
    model, checkpoint = load_projected_predictor(path, args.vjepa_checkpoint, args.radius_px,
                                                 args.patch_coverage_threshold)
    checkpoint_line = describe_checkpoint(path, checkpoint)
    print(f"\nloaded {checkpoint_line}\n  V-JEPA weights: {checkpoint['vjepa_checkpoint_used']}")

    step_seconds = float(checkpoint["step_seconds"])
    frame_stride = frame_stride_for(step_seconds)
    assert frame_stride == checkpoint["frame_stride"], (frame_stride, checkpoint["frame_stride"])
    steps = args.rollout_steps
    tag = model_tag(checkpoint, path)
    figures_dir = args.figures_dir / tag

    dataset = KITTIProjectedPredictorDataset(
        TEST_SEQUENCES,
        context_length=CONTEXT_LENGTH,
        future_length=MAX_STEPS,
        frame_stride=frame_stride,
        image_size=IMAGE_SIZE,
    )
    assert math.isclose(dataset.step_seconds, step_seconds), (dataset.step_seconds, step_seconds)
    print(f"[{tag}] {len(dataset)} test windows: context {CONTEXT_LENGTH} x {step_seconds:g}s, "
          f"{steps} rollout steps of {step_seconds:g}s (horizon {steps * step_seconds:.1f}s)")

    latent_items = range(0, len(dataset), args.latent_every)
    if args.max_windows is not None:
        latent_items = latent_items[:args.max_windows]
    loader = DataLoader(Subset(dataset, latent_items), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)
    print(f"[{tag}] scoring {len(latent_items)} windows in latent space ...")
    latent, agreement = evaluate_latent(model, loader, steps)
    if agreement is not None:
        print(f"[{tag}] model encodings vs cached vjepa_vitb latents (first window): mean |diff| "
              f"{agreement['l1']:.4f} on mean |z| {agreement['target_abs_mean']:.3f}, cosine {agreement['cosine']:.4f}")

    result = {
        "checkpoint": checkpoint_line,
        "checkpoint_path": str(path),
        "vjepa_checkpoint": checkpoint["vjepa_checkpoint_used"],
        "iteration": checkpoint.get("iteration"),
        "val": checkpoint.get("val"),
        "step_seconds": step_seconds,
        "frame_stride": frame_stride,
        "context_length": CONTEXT_LENGTH,
        "horizon_steps": steps,
        "horizon_seconds": [(k + 1) * step_seconds for k in range(steps)],
        "test_sequences": TEST_SEQUENCES,
        "geometry": {"radius_px": args.radius_px, "patch_coverage_threshold": args.patch_coverage_threshold,
                     "image_size": list(IMAGE_SIZE)},
        "latent_every": args.latent_every,
        "target_vs_cache": agreement,
        "latent": latent.summary(),
    }
    if not args.skip_decoded:
        print(f"[{tag}] decoding every {args.decoded_every}th window through the frozen decoders ...")
        result["decoded"] = evaluate_decoded(model, dataset, depth_decoder, semantic_decoder, steps,
                                             args.decoded_every, args.num_workers, args.max_windows)
    print_overview(tag, result)

    figures_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(result, figures_dir / "metrics.json")
    if args.rollouts:
        save_figures(model, dataset, depth_decoder, semantic_decoder, steps, args.rollouts, figures_dir)
        print(f"[{tag}] rollout figures saved to {figures_dir}")
    return tag, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained projected predictors on the test split")
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None,
                        help="projected-predictor checkpoints (default: every "
                             "checkpoints_projected_predictor/projected_predictor_dt*.pt)")
    parser.add_argument("--rollout-steps", type=int, default=MAX_STEPS, choices=range(1, MAX_STEPS + 1),
                        help=f"rollout steps scored per model (the architecture predicts {MAX_STEPS}; "
                             "0.2s model -> 0.8s, 0.5s model -> 2.0s at the default)")
    parser.add_argument("--rollouts", type=int, default=6,
                        help="evenly-spaced test windows to visualise per model (0 disables)")
    parser.add_argument("--latent-every", type=int, default=1,
                        help="score every Nth test window in latent space (1 = all, as the ac evaluator; "
                             "each window is a full model forward)")
    parser.add_argument("--decoded-every", type=int, default=20,
                        help="decode every Nth test window for the decoded-task metrics")
    parser.add_argument("--skip-decoded", action="store_true", help="latent metrics only")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap the windows scored (latent and decoded) -- smoke tests")
    parser.add_argument("--vjepa-checkpoint", type=Path, default=None,
                        help="released V-JEPA 2.1 ViT-B checkpoint (default: the one recorded in the "
                             "model checkpoint if it exists here, else the repo default)")
    parser.add_argument("--radius-px", type=float, default=DEFAULT_RADIUS_PX,
                        help="warp splat radius; must match training (not recorded in the checkpoint)")
    parser.add_argument("--patch-coverage-threshold", type=float, default=DEFAULT_PATCH_COVERAGE_THRESHOLD,
                        help="warp patch validity threshold; must match training (not recorded)")
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
        raise SystemExit(f"no projected-predictor checkpoints found in {PROJECTED_CHECKPOINT_DIR}")

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
    plot_region_curves(results, figures_dir / "latent_l1_by_region.png")
    if any("decoded" in r for r in results.values()):
        plot_decoded_curves(results, figures_dir / "decoded_metrics_vs_horizon.png")
    (figures_dir / "summary.md").write_text(summary_markdown(results))
    write_metrics_json({"models": results}, figures_dir / "metrics.json")
    print(f"\nsummary + comparison plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
