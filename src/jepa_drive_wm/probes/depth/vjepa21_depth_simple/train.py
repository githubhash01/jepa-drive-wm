"""
Train the V-JEPA 2.1 linear depth probe from cached features.

Iteration-based loop (no Lightning). The encoder is offline: we only train the small
linear head + AdaBins depth conversion. Predicted depth is bilinearly upsampled to the
ground-truth resolution before the loss / metrics, so the head stays resolution-agnostic.

Run:
    python -m jepa_drive_wm.probes.depth.vjepa21_depth_simple.train
"""
from __future__ import annotations

import os
from itertools import islice

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DepthProbeConfig
from .dataset import CachedDepthDataset, depth_collate
from .head import DepthProbe
from .losses import build_loss
from .metrics import calculate_depth_metrics


def _infinite(loader):
    while True:
        yield from loader


def build_loaders(cfg: DepthProbeConfig):
    common = dict(
        embeddings_dir=cfg.embeddings_dir,
        kitti_sequences_dir=cfg.kitti_sequences_dir,
        embedding_model=cfg.embedding_model,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        ram_cache=cfg.ram_cache,
    )
    train_ds = CachedDepthDataset(cfg.train_sequences, **common)
    val_ds = CachedDepthDataset(cfg.val_sequences, **common)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=depth_collate, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=depth_collate, pin_memory=True,
    )
    return train_loader, val_loader


def predict_depth(probe: DepthProbe, feat: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Run the probe and upsample the (B,1,gh,gw) depth to the GT resolution (B,1,H,W)."""
    depth = probe(feat)
    return F.interpolate(depth, size=out_hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(probe: DepthProbe, val_loader, cfg: DepthProbeConfig, device, max_batches=None) -> dict:
    probe.eval()
    names = ("a1", "a2", "a3", "abs_rel", "rmse", "silog")
    totals = {n: 0.0 for n in names}
    count = 0
    batches = val_loader if max_batches is None else islice(val_loader, max_batches)
    for feat, depth, valid in batches:
        feat, depth, valid = feat.to(device), depth.to(device), valid.to(device)
        pred = predict_depth(probe, feat, depth.shape[-2:])
        pred = pred.clamp(cfg.min_depth, cfg.max_depth)
        m = calculate_depth_metrics(depth, pred, valid)
        for n in names:
            totals[n] += float(getattr(m, n))
        count += 1
    probe.train()
    return {n: totals[n] / max(count, 1) for n in names}


def train(cfg: DepthProbeConfig | None = None):
    cfg = cfg or DepthProbeConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_loader, val_loader = build_loaders(cfg)
    probe = DepthProbe.from_config(cfg).to(device)
    criterion = build_loss(cfg.losses).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"[train] device={device}  trainable params={n_params:,}  iters={cfg.total_iters}")

    best_abs_rel = float("inf")
    probe.train()
    for it, (feat, depth, valid) in enumerate(islice(_infinite(train_loader), cfg.total_iters), start=1):
        feat, depth, valid = feat.to(device), depth.to(device), valid.to(device)
        pred = predict_depth(probe, feat, depth.shape[-2:])
        loss = criterion(pred, depth, valid)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
        optim.step()

        if it % cfg.log_every == 0:
            print(f"[train] iter {it:>6}/{cfg.total_iters}  loss {loss.item():.4f}")

        if it % cfg.eval_every == 0 or it == cfg.total_iters:
            metrics = evaluate(probe, val_loader, cfg, device)
            msg = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[eval ] iter {it:>6}  {msg}")
            if metrics["abs_rel"] < best_abs_rel:
                best_abs_rel = metrics["abs_rel"]
                ckpt = os.path.join(cfg.out_dir, "depth_probe_best.pt")
                torch.save({"head": probe.state_dict(), "cfg": cfg, "metrics": metrics, "iter": it}, ckpt)
                print(f"[eval ] new best abs_rel={best_abs_rel:.4f} -> saved {ckpt}")

    print(f"[train] done. best abs_rel={best_abs_rel:.4f}")
    return probe


if __name__ == "__main__":
    train()
