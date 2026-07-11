"""Visualisation: RGB | ground-truth | prediction triptychs for depth and semseg."""
from __future__ import annotations

import os

import numpy as np


def save_depth_triptych(rgb, gt_vis, pred, save_path, pred_title, rgb_name=""):
    """3-row RGB / GT depth / predicted depth figure (``gt_vis`` may contain NaNs)."""
    import matplotlib.pyplot as plt

    vmax = float(np.nanpercentile(gt_vis, 95)) if np.isfinite(gt_vis).any() else float(np.nanmax(pred))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    if rgb is not None:
        axes[0].imshow(rgb)
    axes[0].set_title(f"RGB  {rgb_name}".rstrip())
    axes[1].imshow(gt_vis, cmap="plasma", vmin=0, vmax=vmax)
    axes[1].set_title("Ground-truth depth (FoundationStereo)")
    axes[2].imshow(pred, cmap="plasma", vmin=0, vmax=vmax)
    axes[2].set_title(pred_title)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved visualization: {save_path}")


def save_semseg_triptych(rgb, gt_labels, pred_labels, palette, save_path, pred_title,
                         class_names=None, rgb_name=""):
    """3-row RGB / GT overlay / prediction overlay figure. ``palette`` is (C,3) uint8."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    gt_rgb = palette[gt_labels]
    pred_rgb = palette[pred_labels]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    base = rgb if rgb is not None else np.zeros_like(gt_rgb)
    axes[0].imshow(base)
    axes[0].set_title(f"RGB  {rgb_name}".rstrip())
    axes[1].imshow(base); axes[1].imshow(gt_rgb, alpha=0.6)
    axes[1].set_title("Ground-truth segmentation (OneFormer)")
    axes[2].imshow(base); axes[2].imshow(pred_rgb, alpha=0.6)
    axes[2].set_title(pred_title)
    if class_names is not None:
        present = sorted(set(np.unique(gt_labels).tolist()) | set(np.unique(pred_labels).tolist()))
        handles = [Patch(color=palette[i] / 255.0, label=class_names[i]) for i in present]
        axes[2].legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved visualization: {save_path}")
