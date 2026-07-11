"""Visualise the SFP dense probe on the held-out test sequences (depth or semseg).

  --per-seq K : K evenly-spread frames PER test sequence -> test_viz/seq<NN>/
  --index N   : one frame from the combined test set.
"""
from __future__ import annotations

import argparse
import os

import torch

from .config import DenseConfig
from .features import build_wrapper, make_featurizer
from .tasks import get_task
from .train import predict_dense


def _load(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: DenseConfig = blob["cfg"]
    task = get_task(cfg.task)
    model = task.build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    h, w = blob.get("image_hw") or cfg.image_hw or cfg.target_hw
    wrapper = build_wrapper(cfg, (h, w))
    wrapper.verbose = False
    featurize = make_featurizer(wrapper, cfg)
    return cfg, task, model, featurize, device


@torch.no_grad()
def _render(cfg, task, model, featurize, device, ds, idx, save_path):
    x, target, valid = ds[idx]
    feat = featurize(x[None].to(device))
    pred = predict_dense(model, feat, target.shape[-2:])
    task.render(cfg, pred, target[None], valid[None], ds, idx, save_path)


def _spread(n_total, k):
    k = min(k, n_total)
    return [round(i * (n_total - 1) / max(k - 1, 1)) for i in range(k)]


def visualize_spread(ckpt_path: str, per_seq: int = 8, save_root: str | None = None):
    cfg, task, model, featurize, device = _load(ckpt_path)
    save_root = save_root or os.path.join(os.path.dirname(ckpt_path), "test_viz")
    for seq in cfg.test_sequences:
        ds = task.build_dataset(cfg, [seq], augment=False)
        seq_dir = os.path.join(save_root, f"seq{seq:02d}")
        for idx in _spread(len(ds), per_seq):
            frame = os.path.splitext(os.path.basename(ds.samples[idx][0]))[0]
            _render(cfg, task, model, featurize, device, ds, idx, os.path.join(seq_dir, f"frame{frame}.png"))
        print(f"[viz] seq {seq:02d}: {min(per_seq, len(ds))} frames -> {seq_dir}")


def visualize(ckpt_path: str, index: int = 0, save_path: str | None = None):
    cfg, task, model, featurize, device = _load(ckpt_path)
    ds = task.build_dataset(cfg, cfg.test_sequences, augment=False)
    save_path = save_path or os.path.join(os.path.dirname(ckpt_path), f"pred_idx{index:04d}.png")
    _render(cfg, task, model, featurize, device, ds, index, save_path)


def main():
    p = argparse.ArgumentParser(description="Visualise the SFP dense probe on the test set.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--per-seq", type=int, default=None, help="frames per test sequence -> test_viz/seq<NN>/")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()
    if args.per_seq is not None:
        visualize_spread(args.ckpt, args.per_seq, args.save)
    else:
        visualize(args.ckpt, args.index, args.save)


if __name__ == "__main__":
    main()
