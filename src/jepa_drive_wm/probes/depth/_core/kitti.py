"""Shared KITTI depth plumbing for both probes.

The two probes differ only in what the model input is (cached feature .npy vs raw image
re-encoded online). Everything about the *target* -- loading a FoundationStereo depth PNG,
building the valid-pixel mask, resizing both to a fixed ``target_hw`` so mixed-resolution
sequences batch together -- is identical, and lives here. So does the batch collate and
the ImageNet normalisation constants (which must match ``VJEPA21Wrapper`` so online
features equal the cached ones).

Depth = uint16 PNG / 256 -> metres (same recipe as data/kitti.py ``load_depth``).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Must match VJEPA21Wrapper.IMAGENET_MEAN/STD so online-encoded features are identical
# to the cached ones (which were built with the wrapper's preprocessing).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def depth_png_path(sequences_dir: str, seq: int, frame: int) -> str:
    return os.path.join(sequences_dir, f"{seq:02d}", "depth", f"{frame:06d}.png")


def load_depth_metres(path: str) -> np.ndarray:
    """Load a KITTI-style depth PNG as float32 metres (uint16 / 256)."""
    depth = np.asarray(Image.open(path))
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 256.0
    return depth.astype(np.float32, copy=False)


def load_depth_and_mask(depth_path: str, min_depth: float, max_depth: float, target_hw):
    """Depth PNG -> ``(depth, valid)`` tensors of shape (1, H, W), resized to ``target_hw``.

    ``valid`` marks finite pixels inside (min_depth, max_depth). Nearest-neighbour resize
    keeps depth metric and the mask crisp at discontinuities.
    """
    depth = torch.from_numpy(load_depth_metres(depth_path))[None]  # (1, H, W)
    valid = (depth > min_depth) & (depth < max_depth) & torch.isfinite(depth)
    if target_hw is not None and tuple(depth.shape[-2:]) != tuple(target_hw):
        depth = F.interpolate(depth[None].float(), size=target_hw, mode="nearest")[0]
        valid = F.interpolate(valid[None].float(), size=target_hw, mode="nearest")[0] > 0.5
    return depth, valid


def is_usable_depth(depth_path: str) -> bool:
    """A depth PNG exists and is non-empty (some FoundationStereo exports left 0-byte files)."""
    return os.path.exists(depth_path) and os.path.getsize(depth_path) > 0


def depth_collate(batch):
    """Stack a batch. Model inputs share one shape; depth/mask share ``target_hw``.

    Works for either probe: the first element is cached features ((D,gh,gw) or
    (L,D,gh,gw)) or a preprocessed image ((C,H,W)); ``torch.stack`` adds the batch dim.
    """
    feats, depths, valids = zip(*batch)
    depth_shapes = {tuple(d.shape) for d in depths}
    if len(depth_shapes) != 1:
        raise ValueError(
            f"Mixed depth resolutions in one batch: {depth_shapes}. "
            "Set a single config.target_hw so all samples are resized to it."
        )
    return torch.stack(feats), torch.stack(depths), torch.stack(valids)
