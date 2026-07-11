"""Raw KITTI image + depth, for *online* 4-layer feature extraction (no cached .npy).

Returns ``(image_chw, depth, valid)`` where ``image_chw`` is the preprocessed RGB tensor
(resized + ImageNet-normalised, exactly matching ``VJEPA21Wrapper``). The training loop
runs the frozen encoder on the batched images to tap the 4 hierarchical layers, so nothing
is written to disk. Depth/mask handling comes from ``_core.kitti``.
"""
from __future__ import annotations

import glob
import os
from typing import Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from .._core.kitti import IMAGENET_MEAN, IMAGENET_STD, depth_png_path, is_usable_depth, load_depth_and_mask


class ImageDepthDataset(Dataset):
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
        # Train-only augmentations. Photometric ones are geometry-preserving (depth
        # unaffected); horizontal flip is applied jointly to image+depth+mask. NO vertical
        # flip: it breaks monocular depth's vertical prior (near at bottom, far at top).
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
                print(f"[ImageDepthDataset] no images for seq {seq:02d}: {img_dir}")
                continue
            kept = 0
            for img in img_paths:
                frame = int(os.path.basename(img)[:-4])
                dp = depth_png_path(kitti_sequences_dir, seq, frame)
                if not is_usable_depth(dp):
                    continue
                self.samples.append((img, dp))
                kept += 1
            print(f"[ImageDepthDataset] seq {seq:02d}: {kept}/{len(img_paths)} frames have depth")

        if not self.samples:
            raise RuntimeError(f"No frames with both images and depth for sequences {list(sequences)}")
        print(f"[ImageDepthDataset] total samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, depth_path = self.samples[idx]
        pil = Image.open(img_path).convert("RGB")
        if self.augment:
            if bool(torch.rand(()) < 0.5):
                gamma = float(0.7 + 0.8 * torch.rand(()))   # exposure variation, gamma in [0.7, 1.5]
                pil = TF.adjust_gamma(pil, gamma=gamma)
            pil = self.rand_grayscale(pil)
        x = self.transform(pil)  # (C, H, W)

        depth, valid = load_depth_and_mask(depth_path, self.min_depth, self.max_depth, self.target_hw)
        if self.augment and bool(torch.rand(()) < 0.5):
            x = torch.flip(x, dims=[-1])
            depth = torch.flip(depth, dims=[-1])
            valid = torch.flip(valid, dims=[-1])
        return x, depth, valid
