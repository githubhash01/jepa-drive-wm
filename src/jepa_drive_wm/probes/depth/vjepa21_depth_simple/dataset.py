"""
Dataset over cached V-JEPA 2.1 features + FoundationStereo depth.

Each sample is one KITTI frame for which BOTH a cached embedding ``.npy`` and a depth
PNG exist. The embeddings live next to the images, written per-frame by
``utils/vjepa_embeddings_builder.py``:

    sequences/NN/vjepa_vitb/000000.npy   (grid_h*grid_w, embed_dim) fp16
    sequences/NN/vjepa_vitb/_metadata.json

Features are returned at patch-grid resolution (D, gh, gw). Depth and a valid-pixel
mask are resized to a fixed ``target_hw`` (the encoder input size) so that any mix of
KITTI sequences -- whose depth PNGs were exported at different resolutions -- can share
one loader and stay aligned with the prediction. The prediction is upsampled to that
same ``target_hw`` in the training/eval loop.

Depth = uint16 PNG / 256 -> metres (same recipe as data/kitti.py ``load_depth``).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _embedding_dir(sequences_dir: str, seq: int, dirname: str) -> str:
    return os.path.join(sequences_dir, f"{seq:02d}", dirname)


def _depth_png_path(sequences_dir: str, seq: int, frame: int) -> str:
    return os.path.join(sequences_dir, f"{seq:02d}", "depth", f"{frame:06d}.png")


def _read_grid(emb_dir: str) -> tuple[int, int]:
    """(grid_h, grid_w) for an embedding cache, from its ``_metadata.json``."""
    with open(os.path.join(emb_dir, "_metadata.json")) as f:
        layout = json.load(f)["layout"]
    return int(layout["grid_h"]), int(layout["grid_w"])


def load_depth_metres(path: str) -> np.ndarray:
    """Load a KITTI-style depth PNG as float32 metres (uint16 / 256)."""
    depth = np.asarray(Image.open(path))
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 256.0
    return depth.astype(np.float32, copy=False)


class CachedDepthDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[int],
        kitti_sequences_dir: str,
        embedding_dirname: str = "vjepa_vitb",
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        target_hw: Optional[tuple[int, int]] = (384, 1248),
        ram_cache: bool = False,
    ):
        self.kitti_sequences_dir = kitti_sequences_dir
        self.embedding_dirname = embedding_dirname
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_hw = target_hw
        self.ram_cache = ram_cache
        self._feat_cache: dict[int, torch.Tensor] = {}

        # Build the sample list by intersecting cached embeddings with depth PNGs.
        # Each sample carries its grid shape so features reshape correctly.
        self.samples: list[tuple[str, str, int, int]] = []
        for seq in sequences:
            emb_dir = _embedding_dir(kitti_sequences_dir, seq, embedding_dirname)
            npy_paths = sorted(glob.glob(os.path.join(emb_dir, "*.npy")))
            if not npy_paths:
                print(f"[CachedDepthDataset] no embeddings for seq {seq:02d}: {emb_dir}")
                continue
            gh, gw = _read_grid(emb_dir)

            kept = 0
            skipped_empty = 0
            for npy in npy_paths:
                frame = int(os.path.basename(npy)[:-4])  # strip ".npy"
                depth_path = _depth_png_path(kitti_sequences_dir, seq, frame)
                if not os.path.exists(depth_path):
                    continue
                # Some FoundationStereo exports left 0-byte (corrupt) PNGs; skip them.
                if os.path.getsize(depth_path) == 0:
                    skipped_empty += 1
                    continue
                self.samples.append((npy, depth_path, gh, gw))
                kept += 1
            note = f" ({skipped_empty} empty/corrupt skipped)" if skipped_empty else ""
            print(f"[CachedDepthDataset] seq {seq:02d}: {kept}/{len(npy_paths)} frames have depth{note}")

        if not self.samples:
            raise RuntimeError(f"No frames with both features and depth for sequences {list(sequences)}")
        print(f"[CachedDepthDataset] total samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_features(self, idx: int) -> torch.Tensor:
        """Return features as (D, gh, gw) float32."""
        if self.ram_cache and idx in self._feat_cache:
            return self._feat_cache[idx].float()

        npy_path, _, gh, gw = self.samples[idx]
        arr = np.load(npy_path)                                   # (gh*gw, D) fp16
        feat = torch.from_numpy(arr).reshape(gh, gw, -1).permute(2, 0, 1).contiguous()
        if self.ram_cache:
            self._feat_cache[idx] = feat  # keep fp16 to save RAM
        return feat.float()

    def __getitem__(self, idx: int):
        feat = self._load_features(idx)                          # (D, gh, gw) float32
        _, depth_path, _, _ = self.samples[idx]

        depth = torch.from_numpy(load_depth_metres(depth_path))[None]  # (1, H, W)
        valid = (depth > self.min_depth) & (depth < self.max_depth) & torch.isfinite(depth)

        if self.target_hw is not None and tuple(depth.shape[-2:]) != tuple(self.target_hw):
            # Nearest keeps depth values metric and the mask crisp at discontinuities.
            depth = F.interpolate(depth[None].float(), size=self.target_hw, mode="nearest")[0]
            valid = F.interpolate(valid[None].float(), size=self.target_hw, mode="nearest")[0] > 0.5

        return feat, depth, valid


def depth_collate(batch):
    """Stack a batch. Features share one grid; depth/mask share ``target_hw``."""
    feats, depths, valids = zip(*batch)
    depth_shapes = {tuple(d.shape) for d in depths}
    if len(depth_shapes) != 1:
        raise ValueError(
            f"Mixed depth resolutions in one batch: {depth_shapes}. "
            "Set a single config.target_hw so all samples are resized to it."
        )
    return torch.stack(feats), torch.stack(depths), torch.stack(valids)
