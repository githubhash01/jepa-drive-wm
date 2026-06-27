"""
Dataset over cached V-JEPA 2.1 features + FoundationStereo depth.

Each sample is one KITTI frame for which BOTH a cached embedding ``.pt`` and a depth
PNG exist. Features are returned at patch-grid resolution (D, gh, gw); depth and a
valid-pixel mask are returned at the native image resolution. Upsampling the
prediction to the depth resolution happens in the training/eval loop.

Notes
-----
* Frame/depth paths are derived from ``(sequence_nr, frame_idx)`` via the KITTI layout
  (see data/kitti.py), NOT from ``ImageEmbedding.image_path`` which is stale.
* Depth = uint16 PNG / 256 -> metres (same recipe as data/kitti.py ``load_depth``).
* All frames in one dataset must share a native depth resolution (true for any single
  KITTI sequence, and for the default train split seq00+seq02). The collate fn raises
  if you mix resolutions in one loader.
"""
from __future__ import annotations

import glob
import os
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .embeddings import load_datapoint


def _depth_png_path(sequences_dir: str, seq: int, frame: int) -> str:
    return os.path.join(sequences_dir, f"{seq:02d}", "depth", f"{frame:06d}.png")


def _embedding_dir(embeddings_dir: str, seq: int, model: str) -> str:
    return os.path.join(embeddings_dir, f"seq{seq:02d}_{model}")


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
        embeddings_dir: str,
        kitti_sequences_dir: str,
        embedding_model: str = "vjepa21_large",
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        ram_cache: bool = False,
    ):
        self.embeddings_dir = embeddings_dir
        self.kitti_sequences_dir = kitti_sequences_dir
        self.embedding_model = embedding_model
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.ram_cache = ram_cache
        self._feat_cache: dict[int, torch.Tensor] = {}

        # Build the (pt_path, depth_path) list by intersecting cached features with depth.
        self.samples: list[tuple[str, str]] = []
        for seq in sequences:
            emb_dir = _embedding_dir(embeddings_dir, seq, embedding_model)
            pt_paths = sorted(glob.glob(os.path.join(emb_dir, "*.pt")))
            if not pt_paths:
                print(f"[CachedDepthDataset] no embeddings for seq {seq:02d}: {emb_dir}")
                continue
            kept = 0
            skipped_empty = 0
            for pt in pt_paths:
                frame = int(os.path.basename(pt)[:-3])
                depth_path = _depth_png_path(kitti_sequences_dir, seq, frame)
                if not os.path.exists(depth_path):
                    continue
                # Some FoundationStereo exports left 0-byte (corrupt) PNGs; skip them.
                if os.path.getsize(depth_path) == 0:
                    skipped_empty += 1
                    continue
                self.samples.append((pt, depth_path))
                kept += 1
            note = f" ({skipped_empty} empty/corrupt skipped)" if skipped_empty else ""
            print(f"[CachedDepthDataset] seq {seq:02d}: {kept}/{len(pt_paths)} frames have depth{note}")

        if not self.samples:
            raise RuntimeError(f"No frames with both features and depth for sequences {list(sequences)}")
        print(f"[CachedDepthDataset] total samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_features(self, idx: int) -> torch.Tensor:
        """Return features as (D, gh, gw) float32."""
        if self.ram_cache and idx in self._feat_cache:
            return self._feat_cache[idx].float()

        pt_path, _ = self.samples[idx]
        emb = load_datapoint(pt_path).image_embedding
        # (gh*gw, D) -> (gh, gw, D) -> (D, gh, gw)
        feat = emb.features.reshape(emb.grid_h, emb.grid_w, -1).permute(2, 0, 1).contiguous()
        if self.ram_cache:
            self._feat_cache[idx] = feat  # keep bf16 to save RAM
        return feat.float()

    def __getitem__(self, idx: int):
        feat = self._load_features(idx)                       # (D, gh, gw) float32
        _, depth_path = self.samples[idx]

        depth = torch.from_numpy(load_depth_metres(depth_path))[None]  # (1, H, W)
        valid = (depth > self.min_depth) & (depth < self.max_depth) & torch.isfinite(depth)
        return feat, depth, valid


def depth_collate(batch):
    """Stack a batch; require uniform depth resolution (one sequence-resolution per loader)."""
    feats, depths, valids = zip(*batch)
    depth_shapes = {tuple(d.shape) for d in depths}
    if len(depth_shapes) != 1:
        raise ValueError(
            f"Mixed depth resolutions in one batch: {depth_shapes}. "
            "Do not mix KITTI sequences of different native resolution in a single loader."
        )
    return torch.stack(feats), torch.stack(depths), torch.stack(valids)
