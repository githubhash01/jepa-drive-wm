"""Shared depth-visualisation helper: RGB | ground-truth | predicted, stacked.

Both probes' ``visualize.py`` build (rgb, gt, pred) their own way (cached vs online) and
hand off to ``save_depth_triptych`` for the actual figure.
"""
from __future__ import annotations

import os

import numpy as np


def image_path_from_depth(depth_path: str, image_dirname: str = "image_2") -> str:
    return depth_path.replace(os.sep + "depth" + os.sep, os.sep + image_dirname + os.sep)


def save_depth_triptych(rgb, gt_vis, pred, save_path: str, pred_title: str, rgb_name: str = ""):
    """Save a 3-row RGB / GT depth / predicted depth figure. ``gt_vis`` may contain NaNs."""
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
