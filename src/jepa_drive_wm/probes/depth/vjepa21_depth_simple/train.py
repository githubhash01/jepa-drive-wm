"""
Train the V-JEPA 2.1 linear depth probe from cached features.

Iteration-based loop (no Lightning). The encoder is offline: we only train the small
linear head + AdaBins depth conversion. Predicted depth is bilinearly upsampled to the
ground-truth resolution before the loss / metrics, so the head stays resolution-agnostic.

Run:
    python -m jepa_drive_wm.probes.depth.vjepa21_depth_simple.train
"""
from __future__ import annotations

import math
import os
from itertools import islice

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DepthProbeConfig
from .dataset import CachedDepthDataset, ImageDepthDataset, depth_collate
from .head import DepthProbe
from .losses import build_loss
from .metrics import calculate_depth_metrics


def _infinite(loader):
    while True:
        yield from loader


def _warmup_cosine(warmup: int, total: int, min_ratio: float = 0.01):
    """LR multiplier: linear warm-up for ``warmup`` iters, then cosine decay to min_ratio."""
    def lr_lambda(it: int) -> float:
        if it < warmup:
            return (it + 1) / max(warmup, 1)
        progress = (it - warmup) / max(total - warmup, 1)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def build_loaders(cfg: DepthProbeConfig, device):
    """Build train/val loaders and a ``featurize`` callable mapping a batch's first
    element to ``(B, ...)`` features on ``device``.

    "cached": the loader already yields features -> ``featurize`` just moves them to GPU.
    "online": the loader yields preprocessed images -> ``featurize`` runs the frozen
    VJEPA encoder (4-layer tap) on each batch, so nothing is cached to disk.
    """
    if cfg.feature_mode == "online":
        h, w = cfg.target_hw
        common = dict(
            kitti_sequences_dir=cfg.kitti_sequences_dir,
            image_dirname=cfg.image_dirname,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            target_hw=cfg.target_hw,
            image_height=h,
            image_width=w,
        )
        train_ds = ImageDepthDataset(cfg.train_sequences, **common)
        val_ds = ImageDepthDataset(cfg.val_sequences, **common)

        from jepa_drive_wm.utils.vjepa_wrapper import VJEPA21Size, VJEPA21Wrapper
        wrapper = VJEPA21Wrapper(
            size=VJEPA21Size[cfg.vjepa_size], image_height=h, image_width=w, verbose=True,
        )

        def featurize(x: torch.Tensor) -> torch.Tensor:
            return wrapper.extract_hierarchical(x)  # (B,C,H,W) -> (B,L,D,gh,gw) on device
    else:
        common = dict(
            kitti_sequences_dir=cfg.kitti_sequences_dir,
            embedding_dirname=cfg.embedding_dirname,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            target_hw=cfg.target_hw,
            ram_cache=cfg.ram_cache,
        )
        train_ds = CachedDepthDataset(cfg.train_sequences, **common)
        val_ds = CachedDepthDataset(cfg.val_sequences, **common)

        def featurize(x: torch.Tensor) -> torch.Tensor:
            return x.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=depth_collate, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=depth_collate, pin_memory=True,
    )
    return train_loader, val_loader, featurize


def predict_depth(probe: DepthProbe, feat: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Run the probe (bf16 autocast on cuda) and upsample depth to the GT resolution (B,1,H,W).

    The decoder's high-res bin logits dominate memory; bf16 halves that and matches the
    precision the V-JEPA features were encoded at. The loss is computed in fp32 (depth is
    cast back below) for numerical stability.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=feat.is_cuda):
        depth = probe(feat)
    return F.interpolate(depth.float(), size=out_hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(probe: DepthProbe, val_loader, cfg: DepthProbeConfig, device, featurize, max_batches=None) -> dict:
    probe.eval()
    names = ("a1", "a2", "a3", "abs_rel", "rmse", "silog")
    totals = {n: 0.0 for n in names}
    count = 0
    batches = val_loader if max_batches is None else islice(val_loader, max_batches)
    for x, depth, valid in batches:
        feat = featurize(x)
        depth, valid = depth.to(device), valid.to(device)
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

    train_loader, val_loader, featurize = build_loaders(cfg, device)
    probe = DepthProbe.from_config(cfg).to(device)
    criterion = build_loss(cfg.losses).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, _warmup_cosine(cfg.warmup_iters, cfg.total_iters))

    n_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"[train] device={device}  head={cfg.head_type}  feature_mode={cfg.feature_mode}  "
          f"trainable params={n_params:,}  iters={cfg.total_iters}")

    best_abs_rel = float("inf")
    probe.train()
    for it, (x, depth, valid) in enumerate(islice(_infinite(train_loader), cfg.total_iters), start=1):
        feat = featurize(x)
        depth, valid = depth.to(device), valid.to(device)
        pred = predict_depth(probe, feat, depth.shape[-2:])
        loss = criterion(pred, depth, valid)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
        optim.step()
        sched.step()

        if it % cfg.log_every == 0:
            print(f"[train] iter {it:>6}/{cfg.total_iters}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}")

        if it % cfg.eval_every == 0 or it == cfg.total_iters:
            metrics = evaluate(probe, val_loader, cfg, device, featurize)
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
