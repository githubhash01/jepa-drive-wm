"""Visual sanity check for the projected predictor's core context.

Verifies stream building and the general projected-predictor approach:

- use the same data interface as training to fetch one KITTI window;
- build the two staggered streams through the real ``ProjectedPredictor``
  forward pass (zero-init head, so predictions == deterministic proposals);
- visualise the context: both 8-frame streams exactly as the video tokenizer
  sees them, the warped RGB against the real futures, and the per-plane
  tubelet validity that defines the predictor's context/target partition;
- encode everything as training does and show what the latent proposal builds
  deterministically at each horizon, in the global PCA RGB space, alongside
  the oracle-correction magnitude ``||Z* - Z0||``.

Figures are written to ``outputs/quick_check/`` and also shown interactively.

Usage::

    PYTHONPATH=src python -m \
        jepa_drive_wm.models.predictors.projected_predictor.quick_check \
        --sequence 9 --start 355 --dt 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from jepa_drive_wm.data.data_interface_projected_predictor import (
    KITTIProjectedPredictorDataset,
)
from jepa_drive_wm.models.predictors.projected_predictor.projected_predictor import (
    ProjectedPredictor,
    ProjectedPredictorOutput,
)
from jepa_drive_wm.models.predictors.projected_predictor.stream_builder import (
    StreamPair,
    VideoStream,
)
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.train.train_projected_predictor import (
    DEFAULT_VJEPA_CHECKPOINT,
    IMAGE_SIZE,
    build_model,
    dt_to_frame_stride,
)
from jepa_drive_wm.training_utils import autocast
from jepa_drive_wm.viz.visualiser import FVVisualizer

GRID_HW = (IMAGE_SIZE[0] // 16, IMAGE_SIZE[1] // 16)  # (24, 78)

# Which warp validity mask each stream's tubelet planes use, and which
# transition field the future planes supply (see docs/projected_predictor.md §9).
PLANE_ROLES = {
    "stream 1": ("(I_t-3, I_t-2)", "(I_t-1, I_t)", "(W_t+1, W_t+2) → V_t+1", "(W_t+3, W_t+4) → V_t+3"),
    "stream 2": ("(I_t-4, I_t-3)", "(I_t-2, I_t-1)", "(I_t, W_t+1) → V_t", "(W_t+2, W_t+3) → V_t+2"),
}
# Index into streams.warps whose patch_valid masks planes 2 and 3 of each stream.
PLANE_MASK_SOURCES = {"stream 1": (1, 3), "stream 2": (0, 2)}
PLANE_TITLE_COLORS = ("dimgray", "dimgray", "tab:blue", "tab:orange")


def _img(frame: torch.Tensor):
    """Any (H,W,3) or (3,H,W) tensor -> displayable numpy image."""
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.permute(1, 2, 0)
    return frame.detach().float().cpu().clamp(0, 1).numpy()


# ---------------------------------------------------------------------------
# Programmatic checks
# ---------------------------------------------------------------------------


def check_stream_masks(name: str, stream: VideoStream, streams: StreamPair) -> None:
    """Verify token counts, partition, flattening order and per-plane masks."""
    Tp, Hp, Wp = stream.tubelet_patch_valid.shape
    total = Tp * Hp * Wp
    n_ctx, n_tgt = stream.num_context_tokens, stream.num_target_tokens
    assert n_ctx + n_tgt == total, f"{name}: {n_ctx}+{n_tgt} != {total}"

    # The flattened (t, h, w) row-major token order must agree with the masks.
    flat_valid = stream.tubelet_patch_valid.reshape(-1)
    assert bool(flat_valid[stream.masks_x[0]].all()), f"{name}: masks_x order wrong"
    assert bool((~flat_valid[stream.masks_y[0]]).all()), f"{name}: masks_y order wrong"

    # Future planes must carry exactly the documented warp validity masks.
    plane_of_target = stream.masks_y[0] // (Hp * Wp)
    per_plane = torch.bincount(plane_of_target, minlength=Tp)
    warp_a, warp_b = PLANE_MASK_SOURCES[name]
    expected = torch.tensor(
        [
            0,
            0,
            int((~streams.warps[warp_a].patch_valid.bool()).sum()),
            int((~streams.warps[warp_b].patch_valid.bool()).sum()),
        ],
        device=per_plane.device,
    )
    assert torch.equal(per_plane, expected), (
        f"{name}: per-plane target counts {per_plane.tolist()} != "
        f"expected {expected.tolist()}"
    )
    print(
        f"  {name}: context {n_ctx:5d} + target {n_tgt:4d} tokens, "
        f"targets per plane {per_plane.tolist()}  ✓"
    )


def check_zero_init_baseline(output: ProjectedPredictorOutput) -> None:
    """With the zero-init head, predictions must equal the proposals exactly."""
    diff = (output.predictions - output.proposals).abs().max().item()
    assert diff == 0.0, f"zero-init head should give predictions == Z0, max diff {diff}"
    print("  zero-init head: predictions == deterministic proposals (B1)  ✓")


def print_baseline_l1(output: ProjectedPredictorOutput) -> None:
    """Per-horizon latent L1 of B1 warp+copy vs B0 copy-last, split by region.

    The split matters: in geometrically missing regions B1 degenerates to a
    (chained) copy anyway, so the warp's value shows up only on valid patches.
    """
    target = output.target_latents.float()
    b1 = (output.proposals.float() - target).abs().mean(dim=-1)[0]        # (4,N)
    b0 = (output.z_t.float() - target).abs().mean(dim=-1)[0]              # (4,N)
    valid = output.patch_valid[0]                                         # (4,N)

    def region_row(err: torch.Tensor, mask: torch.Tensor) -> str:
        cells = []
        for k in range(err.shape[0]):
            selected = err[k][mask[k]]
            cells.append(f"h{k+1} {selected.mean():.4f}" if selected.numel() else f"h{k+1}    n/a")
        return "  ".join(cells)

    print("  per-horizon latent L1 (lower is better):")
    print("    B1 warp+copy, valid  : " + region_row(b1, valid))
    print("    B0 copy-last, valid  : " + region_row(b0, valid))
    print("    B1 warp+copy, missing: " + region_row(b1, ~valid))
    print("    B0 copy-last, missing: " + region_row(b0, ~valid))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_streams(streams: StreamPair, suptitle: str):
    """Both 8-frame streams exactly as tokenised (masked patches black)."""
    fig, axes = plt.subplots(2, 8, figsize=(26, 3.8), squeeze=False)
    pairs = ((streams.stream_1, "stream 1"), (streams.stream_2, "stream 2"))
    for row, (stream, name) in enumerate(pairs):
        frames = stream.rgb.permute(1, 2, 3, 0)  # (8, H, W, 3)
        for col in range(8):
            ax = axes[row][col]
            ax.imshow(_img(frames[col]))
            ax.set_title(
                f"{name}: {stream.frame_labels[col]}" if col == 0 else stream.frame_labels[col],
                fontsize=9,
                color=PLANE_TITLE_COLORS[col // 2],
            )
            ax.axis("off")
    fig.suptitle(suptitle + " — tubelet pairs share colour", fontsize=12)
    fig.tight_layout()
    return fig


def plot_warp_vs_gt(sample: dict, streams: StreamPair, future_ids: list[int]):
    """Deterministic warps (own-mask, as encoded in image mode) vs real futures."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 3.4), squeeze=False)
    for col, (frame_id, warp) in enumerate(zip(future_ids, streams.warps)):
        axes[0][col].imshow(_img(sample["future_rgb"][col]))
        axes[0][col].set_title(f"GT frame {frame_id}", fontsize=9)
        valid_pct = 100.0 * warp.patch_valid.float().mean().item()
        axes[1][col].imshow(_img(warp.rgb_patch_masked))
        axes[1][col].set_title(f"warp, S valid {valid_pct:.0f}%", fontsize=9)
    for ax, label in zip(axes[:, 0], ("real future", "RGB warp")):
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    for ax in axes.flat:
        if not ax.get_ylabel():
            ax.axis("off")
    fig.suptitle("Deterministic RGB forward projection vs ground truth", fontsize=12)
    fig.tight_layout()
    return fig


def plot_validity_planes(streams: StreamPair):
    """Per-plane tubelet validity: the predictor's context/target partition."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 3.2), squeeze=False)
    pairs = ((streams.stream_1, "stream 1"), (streams.stream_2, "stream 2"))
    for row, (stream, name) in enumerate(pairs):
        for plane in range(4):
            ax = axes[row][plane]
            valid = stream.tubelet_patch_valid[plane]
            ax.imshow(valid.cpu(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            n_tgt = int((~valid).sum())
            ax.set_title(
                f"{PLANE_ROLES[name][plane]}  ({n_tgt} tgt)",
                fontsize=8,
                color=PLANE_TITLE_COLORS[plane],
            )
            ax.set_xticks([]), ax.set_yticks([])
            if plane == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("Tubelet patch validity (white = context token, black = target)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_proposal_pca(
    output: ProjectedPredictorOutput,
    future_ids: list[int],
):
    """GT / proposal / raw warp latents in global PCA space + oracle residual."""
    visualiser = FVVisualizer()
    pca = lambda flat: visualiser.vjepa_embedding_pca(flat.float().cpu(), grid_hw=GRID_HW)

    target = output.target_latents.float()
    residual = (target - output.proposals.float()).norm(dim=-1)  # (1,4,N)

    rows = ("GT  Z*", "proposal  Z0", "warp  Z^warp", "‖Z* − Z0‖")
    fig, axes = plt.subplots(4, 4, figsize=(15, 6.4), squeeze=False)
    for col, frame_id in enumerate(future_ids):
        axes[0][col].imshow(pca(output.target_latents[0, col]), interpolation="nearest")
        axes[1][col].imshow(pca(output.proposals[0, col]), interpolation="nearest")
        axes[2][col].imshow(pca(output.warped_latents[0, col]), interpolation="nearest")
        axes[3][col].imshow(
            residual[0, col].reshape(GRID_HW).cpu(), cmap="magma", interpolation="nearest"
        )
        axes[0][col].set_title(f"t+{col + 1} (frame {frame_id})", fontsize=9)
    for row, label in enumerate(rows):
        for col in range(4):
            axes[row][col].set_xticks([]), axes[row][col].set_yticks([])
        axes[row][0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(
        "Latent proposal in global PCA space — bottom row: where the learned "
        "correction must act",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequence", type=int, default=9)
    parser.add_argument("--start", type=int, default=355, help="raw KITTI start frame")
    parser.add_argument("--dt", type=float, default=0.5, choices=[0.2, 0.5])
    parser.add_argument("--vjepa-checkpoint", type=Path, default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--save-dir", type=Path, default=OUTPUTS_DIR / "quick_check")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_stride = dt_to_frame_stride(args.dt)

    dataset = KITTIProjectedPredictorDataset(
        [args.sequence], frame_stride=frame_stride, image_size=IMAGE_SIZE
    )
    try:
        item = dataset.index.index((args.sequence, args.start))
    except ValueError:
        last = dataset.index[-1][1] if dataset.index else -1
        raise SystemExit(
            f"start={args.start} invalid for sequence {args.sequence}; "
            f"valid range is 0..{last} at stride {frame_stride}."
        )
    sample = dataset[item]
    context_ids = sample["context_frame_ids"].tolist()
    future_ids = sample["future_frame_ids"].tolist()
    print(f"sequence {args.sequence}, dt={args.dt}s (stride {frame_stride})")
    print(f"  context frames {context_ids}, future frames {future_ids}")
    for k, motion in enumerate(sample["ego_motions"], start=1):
        f, r, yaw = motion.tolist()
        print(f"  ego t->t+{k}: forward {f:6.2f} m, right {r:6.2f} m, yaw {yaw:6.3f} rad")

    # Exactly the training construction (zero-init head, frozen everything).
    model: ProjectedPredictor = build_model(
        device=device,
        vjepa_checkpoint=args.vjepa_checkpoint,
        finetune_predictor=False,
    )
    model.eval()
    model.warp_module.K.copy_(sample["intrinsics"].to(model.warp_module.K))

    with autocast(device):
        output = model(
            context_rgb=sample["context_rgb"],
            depth_t=sample["depth_t"],
            ego_motions=sample["ego_motions"],
            future_rgb=sample["future_rgb"],
        )
    streams = output.streams

    print("checks:")
    check_stream_masks("stream 1", streams.stream_1, streams)
    check_stream_masks("stream 2", streams.stream_2, streams)
    check_zero_init_baseline(output)
    print_baseline_l1(output)

    tag = f"seq{args.sequence:02d}_f{context_ids[-1]}_dt{args.dt:g}"
    figures = {
        f"{tag}_streams.png": plot_streams(
            streams, f"Staggered V-JEPA streams (seq {args.sequence}, I_t = frame {context_ids[-1]})"
        ),
        f"{tag}_warp_vs_gt.png": plot_warp_vs_gt(sample, streams, future_ids),
        f"{tag}_validity.png": plot_validity_planes(streams),
        f"{tag}_proposal_pca.png": plot_proposal_pca(output, future_ids),
    }

    args.save_dir.mkdir(parents=True, exist_ok=True)
    for filename, fig in figures.items():
        path = args.save_dir / filename
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"saved {path}")
    plt.show()


if __name__ == "__main__":
    main()
