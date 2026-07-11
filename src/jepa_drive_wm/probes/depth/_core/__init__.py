"""Shared building blocks for the depth probes (see ../README rationale).

Both ``final_probe`` and ``dpt_probe`` import from here; nothing in this package is
probe-specific.

* ``losses``  — SigLoss & friends, ``build_loss``.
* ``metrics`` — ``calculate_depth_metrics`` (a1/a2/a3/abs_rel/rmse/silog).
* ``binning`` — logits -> metric depth (fixed ``FeaturesToDepth``).
* ``kitti``   — depth PNG loading, valid mask, resize, batch collate.
"""
from .binning import FeaturesToDepth
from .kitti import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    depth_collate,
    depth_png_path,
    is_usable_depth,
    load_depth_and_mask,
    load_depth_metres,
)
from .losses import build_loss
from .metrics import calculate_depth_metrics

__all__ = [
    "FeaturesToDepth",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "depth_collate",
    "depth_png_path",
    "is_usable_depth",
    "load_depth_and_mask",
    "load_depth_metres",
    "build_loss",
    "calculate_depth_metrics",
]
