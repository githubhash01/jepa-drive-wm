"""
Visual sanity-check: RGB | ground-truth depth | predicted depth for one frame.

Run:
    python -m jepa_drive_wm.probes.depth.vjepa21_depth_simple.visualize \
        --ckpt /home/hashim/Desktop/Outputs/vjepa21_depth/depth_probe_best.pt --index 0
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .config import DepthProbeConfig
from .dataset import CachedDepthDataset, ImageDepthDataset, load_depth_metres
from .head import DepthProbe
from .train import predict_depth


def _image_path_from_depth(depth_path: str) -> str:
    return depth_path.replace(os.sep + "depth" + os.sep, os.sep + "image_2" + os.sep)


def visualize(ckpt_path: str, index: int = 0, save_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: DepthProbeConfig = blob.get("cfg", DepthProbeConfig())

    probe = DepthProbe.from_config(cfg).to(device)
    probe.load_state_dict(blob["head"])
    probe.eval()

    if cfg.feature_mode == "online":
        h, w = cfg.target_hw
        ds = ImageDepthDataset(
            cfg.val_sequences, cfg.kitti_sequences_dir, cfg.image_dirname,
            cfg.min_depth, cfg.max_depth, cfg.target_hw, image_height=h, image_width=w,
        )
        from jepa_drive_wm.utils.vjepa_wrapper import VJEPA21Size, VJEPA21Wrapper
        wrapper = VJEPA21Wrapper(size=VJEPA21Size[cfg.vjepa_size], image_height=h, image_width=w, verbose=False)

        def featurize(x):
            return wrapper.extract_hierarchical(x)
    else:
        ds = CachedDepthDataset(
            cfg.val_sequences, cfg.kitti_sequences_dir, cfg.embedding_dirname,
            cfg.min_depth, cfg.max_depth, cfg.target_hw,
        )

        def featurize(x):
            return x.to(device)

    x, depth, valid = ds[index]
    depth_path = ds.samples[index][1]

    with torch.no_grad():
        pred = predict_depth(probe, featurize(x[None]), depth.shape[-2:])
        pred = pred.clamp(cfg.min_depth, cfg.max_depth)[0, 0].cpu().numpy()

    gt = depth[0].numpy()
    gt_vis = np.where(valid[0].numpy(), gt, np.nan)

    img_path = _image_path_from_depth(depth_path)
    rgb = np.asarray(Image.open(img_path).convert("RGB")) if os.path.exists(img_path) else None

    vmax = float(np.nanpercentile(gt_vis, 95)) if np.isfinite(gt_vis).any() else cfg.max_depth
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    if rgb is not None:
        axes[0].imshow(rgb)
    axes[0].set_title(f"RGB  ({os.path.basename(img_path)})")
    axes[1].imshow(gt_vis, cmap="plasma", vmin=0, vmax=vmax)
    axes[1].set_title("Ground-truth depth (FoundationStereo)")
    axes[2].imshow(pred, cmap="plasma", vmin=0, vmax=vmax)
    axes[2].set_title("Predicted depth (V-JEPA 2.1 linear probe)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    save_path = save_path or os.path.join(cfg.out_dir, f"depth_pred_idx{index:04d}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved visualization: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()
    visualize(args.ckpt, args.index, args.save)


if __name__ == "__main__":
    main()
