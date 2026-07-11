"""Data plumbing for the dense probe.

Both tasks share the RGB pipeline (resize + ImageNet-normalise, matching ``VJEPA21Wrapper``)
and joint train augmentations; they differ only in the ground truth:
  * ``DepthDataset``  -> FoundationStereo depth PNG (uint16/256 metres) + valid mask.
  * ``SemSegDataset`` -> OneFormer class-id PNG (nearest resize) + all-valid mask.
"""
from __future__ import annotations

import glob
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_depth_metres(path: str) -> np.ndarray:
    depth = np.asarray(Image.open(path))
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 256.0
    return depth.astype(np.float32, copy=False)


def dense_collate(batch):
    """Stack (feature-input image, target, valid); all share ``target_hw``."""
    imgs, targets, valids = zip(*batch)
    shapes = {tuple(t.shape) for t in targets}
    if len(shapes) != 1:
        raise ValueError(f"Mixed target resolutions in one batch: {shapes}. Set a single target_hw.")
    return torch.stack(imgs), torch.stack(targets), torch.stack(valids)


class DenseDataset(Dataset):
    """Base: RGB loading + augmentation. Subclasses provide the GT path + loader."""

    def __init__(self, sequences: Sequence[int], kitti_sequences_dir: str, image_dirname: str = "image_2",
                 target_hw: Optional[tuple[int, int]] = (384, 1248), image_height: int = 384,
                 image_width: int = 1248, augment: bool = False):
        self.kitti_sequences_dir = kitti_sequences_dir
        self.image_dirname = image_dirname
        self.target_hw = target_hw
        self.augment = augment
        self.rand_grayscale = transforms.RandomGrayscale(p=0.2)
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.samples: list[tuple[str, str]] = []
        for seq in sequences:
            img_dir = os.path.join(kitti_sequences_dir, f"{seq:02d}", image_dirname)
            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            if not img_paths:
                print(f"[{type(self).__name__}] no images for seq {seq:02d}: {img_dir}")
                continue
            kept = 0
            for img in img_paths:
                frame = int(os.path.basename(img)[:-4])
                gt = self.gt_path(seq, frame)
                if not (os.path.exists(gt) and os.path.getsize(gt) > 0):
                    continue
                self.samples.append((img, gt))
                kept += 1
            print(f"[{type(self).__name__}] seq {seq:02d}: {kept}/{len(img_paths)} frames have GT")
        if not self.samples:
            raise RuntimeError(f"No frames with both images and GT for sequences {list(sequences)}")
        print(f"[{type(self).__name__}] total samples: {len(self.samples)}")

    # --- subclass hooks -----------------------------------------------------
    def gt_path(self, seq: int, frame: int) -> str:
        raise NotImplementedError

    def load_gt(self, gt_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (target, valid), each (1, H, W) at ``target_hw``."""
        raise NotImplementedError

    # --- shared -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, gt_path = self.samples[idx]
        pil = Image.open(img_path).convert("RGB")
        if self.augment:
            if bool(torch.rand(()) < 0.5):
                gamma = float(0.7 + 0.8 * torch.rand(()))
                pil = TF.adjust_gamma(pil, gamma=gamma)
            pil = self.rand_grayscale(pil)
        x = self.transform(pil)
        target, valid = self.load_gt(gt_path)
        if self.augment and bool(torch.rand(()) < 0.5):
            x = torch.flip(x, dims=[-1])
            target = torch.flip(target, dims=[-1])
            valid = torch.flip(valid, dims=[-1])
        return x, target, valid


class DepthDataset(DenseDataset):
    def __init__(self, *args, min_depth=0.001, max_depth=80.0, **kwargs):
        self.min_depth = min_depth
        self.max_depth = max_depth
        super().__init__(*args, **kwargs)

    def gt_path(self, seq: int, frame: int) -> str:
        return os.path.join(self.kitti_sequences_dir, f"{seq:02d}", "depth", f"{frame:06d}.png")

    def load_gt(self, gt_path: str):
        depth = torch.from_numpy(load_depth_metres(gt_path))[None]  # (1, H, W)
        valid = (depth > self.min_depth) & (depth < self.max_depth) & torch.isfinite(depth)
        if self.target_hw is not None and tuple(depth.shape[-2:]) != tuple(self.target_hw):
            depth = F.interpolate(depth[None].float(), size=self.target_hw, mode="nearest")[0]
            valid = F.interpolate(valid[None].float(), size=self.target_hw, mode="nearest")[0] > 0.5
        return depth, valid


class SemSegDataset(DenseDataset):
    def __init__(self, *args, semantic_root: str, **kwargs):
        self.semantic_root = semantic_root
        super().__init__(*args, **kwargs)

    def gt_path(self, seq: int, frame: int) -> str:
        return os.path.join(self.semantic_root, f"{seq:02d}", f"{frame:06d}.png")

    def load_gt(self, gt_path: str):
        labels = torch.from_numpy(np.asarray(Image.open(gt_path)).astype(np.int64))[None]  # (1, H, W)
        if self.target_hw is not None and tuple(labels.shape[-2:]) != tuple(self.target_hw):
            labels = F.interpolate(labels[None].float(), size=self.target_hw, mode="nearest")[0].long()
        valid = torch.ones_like(labels, dtype=torch.bool)
        return labels, valid
