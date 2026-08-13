"""
Experiment: Geometric transport of V-JEPA features

Question
--------
Can known depth, camera intrinsics, and future ego-motion transform the
current observation into an anchor that is closer to the true future
V-JEPA representation than the unmodified current representation?

Target
------
z_future = VJEPA(rgb_future)

Methods
-------
1. Copy baseline:
       anchor_copy = z_current

2. RGB transport:
       rgb_warped = project_splat(
           rgb_current,
           depth_current,
           intrinsics,
           T_future_from_current,
       )
       anchor_rgb = VJEPA(rgb_warped)

3. Latent transport:
       patch_depth = median depth within each V-JEPA patch
       anchor_latent = project_splat(
           z_current,
           patch_depth,
           patch_intrinsics,
           T_future_from_current,
       )

Evaluation
----------
Compare every anchor with z_future on the same valid, geometrically
transportable target patches.

Primary metric:
    mean per-patch cosine distance

Supporting metric:
    per-patch L1 distance

Also Report:
    overall L1 distance


Visualise the Error maps, and PCA of predicted vs true VJEPA
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from jepa_drive_wm.data.kitti import KITTISequence
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.utils.vjepa_wrapper import VJEPA21Size, VJEPA21Wrapper
from jepa_drive_wm.world_model.geometric.warp import (
    fill_holes,
    forward_splat,
    forward_splat_soft,
    patch_intrinsics,
    scale_intrinsics,
)

FIGURES_DIR = OUTPUTS_DIR / "geometric_transport"

# A target patch only enters the evaluation if BOTH transports could reach it:
# the latent splat put a token there, and the RGB splat covered enough of its
# pixels that the re-encoded feature is not dominated by hole filling.
MIN_RGB_COVERAGE = 0.5
# A source patch contributes to latent transport only if enough of its pixels
# carry valid depth for the median to mean anything.
MIN_PATCH_DEPTH_FRAC = 0.3

IMAGENET_MEAN = torch.tensor(VJEPA21Wrapper.IMAGENET_MEAN)


@dataclass(frozen=True)
class Example:
    seq_nr: int
    t: int
    dt: int

    @property
    def name(self) -> str:
        return f"seq{self.seq_nr:02d}_t{self.t:06d}_dt{self.dt:02d}"


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

def patch_median_depth(
    depth_vjepa: torch.Tensor, grid_hw: tuple[int, int], patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Median valid depth per V-JEPA patch. Returns (depth (gh, gw), valid_frac (gh, gw))."""
    gh, gw = grid_hw
    blocks = (
        depth_vjepa.reshape(gh, patch_size, gw, patch_size)
        .permute(0, 2, 1, 3)
        .reshape(gh, gw, patch_size * patch_size)
    )
    valid = blocks > 0
    frac = valid.float().mean(dim=-1)
    med = torch.where(valid, blocks, torch.nan).nanmedian(dim=-1).values
    return torch.nan_to_num(med, nan=0.0), frac


@torch.no_grad()
def compute_anchors(seq: KITTISequence, wrapper: VJEPA21Wrapper, t: int, dt: int) -> dict:
    """Build the three anchors (+ fresh-encode diagnostics) for one (t, t+dt) pair."""
    layout = wrapper.layout(num_frames=1)
    grid_hw = (layout.grid_h, layout.grid_w)
    assert tuple(seq.vjepa_grid_hw) == grid_hw, (
        f"cache grid {seq.vjepa_grid_hw} != wrapper grid {grid_hw}"
    )
    vjepa_hw = (wrapper.img_height, wrapper.img_width)

    rgb_current = torch.from_numpy(seq.get_image(t))            # (H, W, 3) in [0, 1]
    rgb_future = torch.from_numpy(seq.get_image(t + dt))
    depth_current = torch.from_numpy(seq.get_depth(t))          # (H, W) metres, c2
    native_hw = depth_current.shape
    assert rgb_current.shape[:2] == native_hw

    z_current = torch.from_numpy(seq.get_vjepa_features(t, as_grid=True))
    z_future = torch.from_numpy(seq.get_vjepa_features(t + dt, as_grid=True))

    # Everything geometric happens in c2; get_camera_se3(t, t+dt) is exactly
    # T_future_from_current for left-camera points.
    T_fut_cur = seq.get_camera_se3(t, t + dt)
    K_native = seq.calib.K2

    # ---- method 2: transport RGB at native resolution, then re-encode -------
    rgb_warped, rgb_valid, rgb_depth_dst = forward_splat(
        rgb_current, depth_current, K_native, T_fut_cur
    )
    rgb_filled = fill_holes(rgb_warped, rgb_valid, iterations=8, fill_value=IMAGENET_MEAN)
    rgb_filled = rgb_filled.clamp(0.0, 1.0)

    # ---- method 3: transport tokens on the patch lattice --------------------
    # Depth -> V-JEPA input resolution (nearest: no blending across edges),
    # then median per 16x16 patch.
    depth_vjepa = F.interpolate(
        depth_current[None, None], size=vjepa_hw, mode="nearest"
    )[0, 0]
    p_depth, p_frac = patch_median_depth(depth_vjepa, grid_hw, layout.patch_size)

    K_vjepa = scale_intrinsics(
        K_native, vjepa_hw[1] / native_hw[1], vjepa_hw[0] / native_hw[0]
    )
    K_patch = patch_intrinsics(K_vjepa, layout.patch_size)

    latent_args = (z_current, p_depth, K_patch, T_fut_cur)
    latent_kwargs = {"src_valid": p_frac >= MIN_PATCH_DEPTH_FRAC}
    anchor_latent, latent_valid, _ = forward_splat(*latent_args, **latent_kwargs)
    # Nearest-cell splatting aliases on the coarse token lattice (forward motion
    # expands the scene, so winners spread apart leaving regular holes); the
    # soft variant spreads each token over its 4 neighbours instead.
    anchor_latent_soft, latent_soft_valid, _ = forward_splat_soft(
        *latent_args, **latent_kwargs
    )

    # ---- one encoder batch: warped RGB + fresh current/future diagnostics ---
    def to_uint8(img: torch.Tensor) -> np.ndarray:
        return (img.numpy() * 255.0).round().astype(np.uint8)

    tokens = wrapper.encode_images(
        [to_uint8(rgb_filled), to_uint8(rgb_current), to_uint8(rgb_future)]
    )
    grids = wrapper.unflatten(tokens, num_frames=1)[:, 0]      # (3, gh, gw, D)
    anchor_rgb, z_current_fresh, z_future_fresh = grids[0], grids[1], grids[2]

    # ---- masks ---------------------------------------------------------------
    # RGB-warp coverage per target patch, measured at the encoder's resolution.
    rgb_valid_vjepa = F.interpolate(
        rgb_valid[None, None].float(), size=vjepa_hw, mode="nearest"
    )
    coverage = F.avg_pool2d(rgb_valid_vjepa, layout.patch_size)[0, 0]
    eval_mask = latent_valid & (coverage >= MIN_RGB_COVERAGE)

    return {
        "rgb_current": rgb_current,
        "rgb_future": rgb_future,
        "rgb_warped": rgb_warped,
        "rgb_filled": rgb_filled,
        "rgb_valid": rgb_valid,
        "rgb_depth_dst": rgb_depth_dst,
        "coverage": coverage,
        "latent_valid": latent_valid,
        "latent_soft_valid": latent_soft_valid,
        "eval_mask": eval_mask,
        "z_current": z_current,
        "z_future": z_future,
        "anchors": {
            "copy": z_current,
            "rgb_warp": anchor_rgb,
            "latent_warp": anchor_latent,
            "latent_soft": anchor_latent_soft,
            "reencode_copy": z_current_fresh,   # diagnostic: copy without cache noise
            "noise_floor": z_future_fresh,      # diagnostic: fresh future vs cached future
        },
        "displacement_m": float(torch.linalg.norm(T_fut_cur[:3, 3])),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compare(anchor: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """Per-patch error maps + scalars over the masked patches."""
    cos_map = 1.0 - F.cosine_similarity(anchor, target, dim=-1)   # (gh, gw)
    l1_map = (anchor - target).abs().mean(dim=-1)                 # (gh, gw)
    return {
        "cos_map": cos_map,
        "l1_map": l1_map,
        "cos_dist": cos_map[mask].mean().item(),
        "l1_per_patch": (anchor - target).abs().sum(dim=-1)[mask].mean().item(),
        "l1_overall": (anchor[mask] - target[mask]).abs().mean().item(),
    }


def evaluate(result: dict) -> dict[str, dict]:
    mask = result["eval_mask"]
    return {
        name: compare(anchor, result["z_future"], mask)
        for name, anchor in result["anchors"].items()
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def joint_pca_rgb(
    grids: list[torch.Tensor], masks: list[torch.Tensor]
) -> list[np.ndarray]:
    """Project all grids through ONE shared 3-component PCA so colours are
    comparable between predicted and true features. Invalid patches -> grey."""
    X = torch.cat([g[m] for g, m in zip(grids, masks)]).double().numpy()
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    components = Vt[:3]

    projections = [((g.double().numpy() - mean) @ components.T) for g in grids]
    stacked = np.concatenate(
        [p[m.numpy()] for p, m in zip(projections, masks)], axis=0
    )
    lo = np.percentile(stacked, 1, axis=0)
    hi = np.percentile(stacked, 99, axis=0)
    hi = np.where(hi > lo, hi, lo + 1e-6)

    images = []
    for proj, mask in zip(projections, masks):
        img = np.clip((proj - lo) / (hi - lo), 0.0, 1.0)
        img[~mask.numpy()] = 0.35
        images.append(img)
    return images


def visualise(example: Example, result: dict, metrics: dict) -> None:
    mask = result["eval_mask"]
    full = torch.ones_like(mask)
    pca_imgs = joint_pca_rgb(
        [result["z_current"], result["z_future"], result["anchors"]["rgb_warp"],
         result["anchors"]["latent_warp"], result["anchors"]["latent_soft"]],
        [full, full, full, result["latent_valid"], result["latent_soft_valid"]],
    )

    methods = ["copy", "rgb_warp", "latent_warp", "latent_soft"]
    masked_map = lambda m, kind: np.ma.masked_where(~mask.numpy(), metrics[m][kind].numpy())
    cos_vmax = max(np.percentile(metrics[m]["cos_map"][mask], 98) for m in methods)
    l1_vmax = max(np.percentile(metrics[m]["l1_map"][mask], 98) for m in methods)
    err_cmap = plt.cm.magma.copy()
    err_cmap.set_bad(color="0.6")

    fig, axes = plt.subplots(4, 5, figsize=(28, 10.5))
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])

    row = axes[0]
    row[0].imshow(result["rgb_current"]); row[0].set_title(f"current (t={example.t})")
    row[1].imshow(result["rgb_future"]); row[1].set_title(f"future (t+{example.dt})")
    row[2].imshow(result["rgb_filled"]); row[2].set_title("warped RGB (V-JEPA input)")
    row[3].imshow(result["rgb_valid"], cmap="gray"); row[3].set_title("splat coverage (native)")
    row[4].imshow(np.ma.masked_where(~result["rgb_valid"].numpy(),
                                     result["rgb_depth_dst"].numpy()), cmap="plasma")
    row[4].set_title("warped depth (native)")

    row = axes[1]
    titles = ["PCA z_current", "PCA z_future (target)", "PCA anchor_rgb",
              "PCA anchor_latent (nearest)", "PCA anchor_latent (soft)"]
    for ax, img, title in zip(row, pca_imgs, titles):
        ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_title(title)

    for ax, m in zip(axes[2], methods):
        im_cos = ax.imshow(masked_map(m, "cos_map"), cmap=err_cmap, vmin=0,
                           vmax=cos_vmax, interpolation="nearest", aspect="auto")
        ax.set_title(f"cos dist: {m} ({metrics[m]['cos_dist']:.3f})")
    axes[2, 4].imshow(mask, cmap="gray", interpolation="nearest", aspect="auto")
    axes[2, 4].set_title(f"eval mask ({int(mask.sum())}/{mask.numel()} patches)")
    fig.colorbar(im_cos, ax=axes[2, :4], fraction=0.015, pad=0.01)

    for ax, m in zip(axes[3], methods):
        im_l1 = ax.imshow(masked_map(m, "l1_map"), cmap=err_cmap, vmin=0,
                          vmax=l1_vmax, interpolation="nearest", aspect="auto")
        ax.set_title(f"L1 (per dim): {m} ({metrics[m]['l1_overall']:.3f})")
    axes[3, 4].imshow(result["coverage"], cmap="viridis", vmin=0, vmax=1,
                      interpolation="nearest", aspect="auto")
    axes[3, 4].set_title("RGB coverage per patch")
    fig.colorbar(im_l1, ax=axes[3, :4], fraction=0.015, pad=0.01)

    fig.suptitle(
        f"{example.name}  |  displacement {result['displacement_m']:.2f} m  |  "
        + "   ".join(f"{m}: cos {metrics[m]['cos_dist']:.3f}" for m in methods),
        fontsize=14,
    )
    out_path = FIGURES_DIR / f"{example.name}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def scalar_metrics(metrics: dict) -> dict:
    return {
        name: {k: v for k, v in m.items() if not torch.is_tensor(v)}
        for name, m in metrics.items()
    }


def print_table(rows: list[tuple[Example, dict, float]]) -> None:
    methods = ["copy", "rgb_warp", "latent_warp", "latent_soft", "noise_floor"]
    header = f"{'example':<24}{'disp[m]':>8} | " + " | ".join(f"{m:>16}" for m in methods)
    print("\n" + header)
    print("-" * len(header))
    print(f"{'':<24}{'':>8} | " + " | ".join(f"{'cos':>8}{'L1':>8}" for _ in methods))
    for example, metrics, disp in rows:
        cells = " | ".join(
            f"{metrics[m]['cos_dist']:>8.4f}{metrics[m]['l1_overall']:>8.4f}"
            for m in methods
        )
        print(f"{example.name:<24}{disp:>8.2f} | {cells}")

    print("-" * len(header))
    for dt in sorted({e.dt for e, _, _ in rows}):
        subset = [m for e, m, _ in rows if e.dt == dt]
        cells = " | ".join(
            f"{np.mean([m[meth]['cos_dist'] for m in subset]):>8.4f}"
            f"{np.mean([m[meth]['l1_overall'] for m in subset]):>8.4f}"
            for meth in methods
        )
        print(f"{f'mean over dt={dt}':<24}{'':>8} | {cells}")


def parse_examples(pairs: list[str], dts: list[int]) -> list[Example]:
    examples = []
    for pair in pairs:
        seq_nr, t = (int(x) for x in pair.split(":"))
        examples.extend(Example(seq_nr, t, dt) for dt in dts)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--examples", nargs="+", default=["07:250", "10:100", "10:600"],
        metavar="SEQ:FRAME", help="sequence:frame pairs to transport from",
    )
    parser.add_argument(
        "--dts", nargs="+", type=int, default=[2, 5, 10],
        help="frame offsets to transport across (10 Hz: 5 = 0.5 s)",
    )
    args = parser.parse_args()

    examples = parse_examples(args.examples, args.dts)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    wrapper = VJEPA21Wrapper(size=VJEPA21Size.BASE, verbose=False)
    sequences: dict[int, KITTISequence] = {}

    rows = []
    for example in examples:
        if example.seq_nr not in sequences:
            sequences[example.seq_nr] = KITTISequence(example.seq_nr)
        seq = sequences[example.seq_nr]
        assert example.t + example.dt < len(seq)

        print(f"{example.name} ...")
        result = compute_anchors(seq, wrapper, example.t, example.dt)
        metrics = evaluate(result)
        visualise(example, result, metrics)
        rows.append((example, metrics, result["displacement_m"]))

    print_table(rows)

    payload = [
        {"example": vars(e) | {"displacement_m": d}, "metrics": scalar_metrics(m)}
        for e, m, d in rows
    ]
    metrics_path = FIGURES_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nmetrics -> {metrics_path}")


if __name__ == "__main__":
    main()
