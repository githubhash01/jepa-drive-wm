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
from torchvision import transforms
import torchvision.transforms.functional as TF

# Must match VJEPA21Wrapper.IMAGENET_MEAN/STD so online-encoded features are identical
# to the cached ones (which were built with the wrapper's preprocessing).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_path(sequences_dir: str, seq: int, frame: int, image_dirname: str) -> str:
    return os.path.join(sequences_dir, f"{seq:02d}", image_dirname, f"{frame:06d}.png")


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
        """Return features as (D, gh, gw), or (L, D, gh, gw) for a hierarchical cache.

        A 2-D cache array ``(gh*gw, D)`` is the legacy single (final) layer. A 3-D array
        ``(L, gh*gw, D)`` is the 4-layer hierarchical cache written with
        ``--hierarchical`` (one grid per tapped layer), used by the DPT head.
        """
        if self.ram_cache and idx in self._feat_cache:
            return self._feat_cache[idx].float()

        npy_path, _, gh, gw = self.samples[idx]
        arr = np.load(npy_path)                                   # (gh*gw, D) or (L, gh*gw, D) fp16
        t = torch.from_numpy(arr)
        if t.ndim == 3:
            # (L, gh*gw, D) -> (L, D, gh, gw)
            L = t.shape[0]
            feat = t.reshape(L, gh, gw, -1).permute(0, 3, 1, 2).contiguous()
        else:
            # (gh*gw, D) -> (D, gh, gw)
            feat = t.reshape(gh, gw, -1).permute(2, 0, 1).contiguous()
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


class ImageDepthDataset(Dataset):
    """Raw KITTI image + depth, for *online* feature extraction (no cached .npy).

    Returns ``(image_chw, depth, valid)`` where ``image_chw`` is the preprocessed RGB
    tensor (resized + ImageNet-normalised, exactly matching ``VJEPA21Wrapper``). The
    training loop runs the frozen encoder on the batched images to get the 4-layer
    features, so nothing is written to disk. Depth/mask handling mirrors
    ``CachedDepthDataset`` so the two modes are interchangeable.
    """

    def __init__(
        self,
        sequences: Sequence[int],
        kitti_sequences_dir: str,
        image_dirname: str = "image_2",
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        target_hw: Optional[tuple[int, int]] = (384, 1248),
        image_height: int = 384,
        image_width: int = 1248,
        augment: bool = False,
    ):
        self.kitti_sequences_dir = kitti_sequences_dir
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_hw = target_hw
        self.augment = augment
        # Train-only augmentations. All photometric ones are geometry-preserving (depth
        # values unaffected); horizontal flip is applied jointly to image+depth+mask.
        # NO vertical flip: it would break monocular depth's vertical prior (near at the
        # bottom, far at the top). Photometric set: random gamma + random grayscale drop.
        self.rand_grayscale = transforms.RandomGrayscale(p=0.2)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_height, image_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        # Intersect images with depth PNGs, frame by frame.
        self.samples: list[tuple[str, str]] = []
        for seq in sequences:
            img_dir = os.path.join(kitti_sequences_dir, f"{seq:02d}", image_dirname)
            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            if not img_paths:
                print(f"[ImageDepthDataset] no images for seq {seq:02d}: {img_dir}")
                continue
            kept = 0
            skipped_empty = 0
            for img in img_paths:
                frame = int(os.path.basename(img)[:-4])
                depth_path = _depth_png_path(kitti_sequences_dir, seq, frame)
                if not os.path.exists(depth_path):
                    continue
                if os.path.getsize(depth_path) == 0:
                    skipped_empty += 1
                    continue
                self.samples.append((img, depth_path))
                kept += 1
            note = f" ({skipped_empty} empty/corrupt skipped)" if skipped_empty else ""
            print(f"[ImageDepthDataset] seq {seq:02d}: {kept}/{len(img_paths)} frames have depth{note}")

        if not self.samples:
            raise RuntimeError(f"No frames with both images and depth for sequences {list(sequences)}")
        print(f"[ImageDepthDataset] total samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, depth_path = self.samples[idx]
        pil = Image.open(img_path).convert("RGB")
        if self.augment:
            # Random gamma (exposure variation): gamma in [0.7, 1.5], applied ~half the time.
            if bool(torch.rand(()) < 0.5):
                gamma = float(0.7 + 0.8 * torch.rand(()))
                pil = TF.adjust_gamma(pil, gamma=gamma)
            pil = self.rand_grayscale(pil)  # drop colour with p=0.2
        x = self.transform(pil)  # (C, H, W)

        depth = torch.from_numpy(load_depth_metres(depth_path))[None]  # (1, H, W)
        valid = (depth > self.min_depth) & (depth < self.max_depth) & torch.isfinite(depth)
        if self.target_hw is not None and tuple(depth.shape[-2:]) != tuple(self.target_hw):
            depth = F.interpolate(depth[None].float(), size=self.target_hw, mode="nearest")[0]
            valid = F.interpolate(valid[None].float(), size=self.target_hw, mode="nearest")[0] > 0.5

        if self.augment and bool(torch.rand(()) < 0.5):
            # Horizontal flip image, depth and mask together (last dim = width).
            x = torch.flip(x, dims=[-1])
            depth = torch.flip(depth, dims=[-1])
            valid = torch.flip(valid, dims=[-1])
        return x, depth, valid


def depth_collate(batch):
    """Stack a batch. Features share one grid; depth/mask share ``target_hw``.

    Works for both modes: the first element is cached features ((D,gh,gw) or
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
