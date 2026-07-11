"""Visual sanity-check for the DPT probe: RGB | GT | predicted depth.

Re-encodes each frame online (like training) at the checkpoint's resolution.

Two modes:
  * ``--index N``       : one frame from the combined test set (quick single check).
  * ``--per-seq K``     : K evenly-spread frames PER held-out test sequence, saved into
                          one folder per sequence (``test_viz/seq10/``, ``test_viz/seq13/``...).

Run:
    python -m jepa_drive_wm.probes.depth.dpt_probe.visualize \
        --ckpt ~/Desktop/Outputs/vjepa21_depth_dpt_metric/final/depth_probe_best.pt --per-seq 8
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .._core.viz import image_path_from_depth, save_depth_triptych
from .config import DPTProbeConfig
from .dataset import ImageDepthDataset
from .head import build_probe
from .train import predict_depth

_LABEL = {"quad": "4-layer", "final": "final-layer", "pred": "predictor-384"}


def _load_probe(ckpt_path: str):
    """Load probe + cfg + frozen encoder wrapper at the checkpoint's input resolution."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: DPTProbeConfig = blob.get("cfg", DPTProbeConfig())
    probe = build_probe(cfg).to(device)
    probe.load_state_dict(blob["head"])
    probe.eval()
    # Input res: the checkpoint's training resolution if recorded, else cfg.image_hw/target_hw.
    h, w = blob.get("image_hw") or cfg.image_hw or cfg.target_hw
    from jepa_drive_wm.utils.vjepa_wrapper import VJEPA21Size, VJEPA21Wrapper
    wrapper = VJEPA21Wrapper(size=VJEPA21Size[cfg.vjepa_size], image_height=h, image_width=w, verbose=False)
    return probe, wrapper, cfg, device, (h, w)


def _featurize(wrapper, cfg, x):
    """One frame -> features for the head (matching training): (1,4,D,gh,gw) for DPT, or
    the single final grid (1,D,gh,gw) for the linear head."""
    feat = wrapper.extract_hierarchical(x[None])              # (1, 4, 768, gh, gw)
    if getattr(cfg, "head_type", "dpt") == "dpt" and cfg.layer_mode == "quad":
        return feat
    final = feat[:, -1]                                       # (1, D, gh, gw) final-normed
    if cfg.layer_mode == "pred":
        final = wrapper.compress_final(final)                 # (1, 384, gh, gw)
    if getattr(cfg, "head_type", "dpt") == "linear":
        return final
    return final.unsqueeze(1).repeat(1, 4, 1, 1, 1)


@torch.no_grad()
def _render_frame(probe, wrapper, cfg, ds, index: int, save_path: str):
    """Render one RGB|GT|pred triptych for ``ds[index]`` and save it."""
    from PIL import Image
    x, depth, valid = ds[index]
    depth_path = ds.samples[index][1]
    feat = _featurize(wrapper, cfg, x)
    pred = predict_depth(probe, feat, depth.shape[-2:]).clamp(cfg.min_depth, cfg.max_depth)
    pred = pred[0, 0].cpu().numpy()

    gt_vis = np.where(valid[0].numpy(), depth[0].numpy(), np.nan)
    img_path = image_path_from_depth(depth_path, cfg.image_dirname)
    rgb = np.asarray(Image.open(img_path).convert("RGB")) if os.path.exists(img_path) else None
    save_depth_triptych(
        rgb, gt_vis, pred, save_path,
        pred_title=f"Predicted depth (V-JEPA 2.1 {_LABEL[cfg.layer_mode]} DPT probe)",
        rgb_name=os.path.basename(img_path),
    )


def _spread_indices(n_total: int, k: int) -> list[int]:
    """``k`` evenly-spread indices over ``[0, n_total-1]`` (clamped to n_total)."""
    k = min(k, n_total)
    return [round(i * (n_total - 1) / max(k - 1, 1)) for i in range(k)]


def visualize(ckpt_path: str, index: int = 0, save_path: str | None = None):
    """Single-frame check on the combined test set (kept for quick spot-checks)."""
    probe, wrapper, cfg, _, (h, w) = _load_probe(ckpt_path)
    ds = ImageDepthDataset(
        cfg.test_sequences, cfg.kitti_sequences_dir, cfg.image_dirname,
        cfg.min_depth, cfg.max_depth, cfg.target_hw, image_height=h, image_width=w,
    )
    # Default: write next to the checkpoint (scheme-agnostic).
    save_path = save_path or os.path.join(os.path.dirname(ckpt_path), f"depth_pred_idx{index:04d}.png")
    _render_frame(probe, wrapper, cfg, ds, index, save_path)


def visualize_spread(ckpt_path: str, per_seq: int = 8, save_root: str | None = None):
    """One folder per held-out test sequence, each with ``per_seq`` evenly-spread depth maps."""
    probe, wrapper, cfg, _, (h, w) = _load_probe(ckpt_path)
    save_root = save_root or os.path.join(os.path.dirname(ckpt_path), "test_viz")
    for seq in cfg.test_sequences:
        ds = ImageDepthDataset(
            [seq], cfg.kitti_sequences_dir, cfg.image_dirname,
            cfg.min_depth, cfg.max_depth, cfg.target_hw, image_height=h, image_width=w,
        )
        seq_dir = os.path.join(save_root, f"seq{seq:02d}")
        for idx in _spread_indices(len(ds), per_seq):
            frame = os.path.splitext(os.path.basename(ds.samples[idx][1]))[0]  # KITTI frame id
            _render_frame(probe, wrapper, cfg, ds, idx, os.path.join(seq_dir, f"frame{frame}.png"))
        print(f"[viz] seq {seq:02d}: {min(per_seq, len(ds))} frames -> {seq_dir}")


def main():
    p = argparse.ArgumentParser(description="Visualize the DPT depth probe on the held-out test set.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--per-seq", type=int, default=None,
                   help="frames per test sequence, saved to test_viz/seq<NN>/ (per-sequence mode)")
    p.add_argument("--index", type=int, default=0, help="single-frame mode: index into the combined test set")
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()
    if args.per_seq is not None:
        visualize_spread(args.ckpt, args.per_seq, args.save)
    else:
        visualize(args.ckpt, args.index, args.save)


if __name__ == "__main__":
    main()
