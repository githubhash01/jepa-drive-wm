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
from .losses import build_loss, chamfer_bin_loss
from .metrics import calculate_depth_metrics


def _infinite(loader):
    while True:
        yield from loader


def _stage_lr(peak_lr: float, warmup_updates: int, total_updates: int, u: int, min_ratio: float = 0.01) -> float:
    """Absolute LR for optimiser update ``u`` within a stage: linear warm-up to ``peak_lr``,
    then cosine decay to ``peak_lr * min_ratio``. Used per-stage so each curriculum phase
    can have its own peak LR (e.g. a lower LR for the high-res fine-tune phase)."""
    if u < warmup_updates:
        return peak_lr * (u + 1) / max(warmup_updates, 1)
    progress = (u - warmup_updates) / max(total_updates - warmup_updates, 1)
    return peak_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_loaders(cfg: DepthProbeConfig, device, image_hw=None):
    """Build train/val loaders and a ``featurize`` callable mapping a batch's first
    element to ``(B, ...)`` features on ``device``.

    "cached": the loader already yields features -> ``featurize`` just moves them to GPU.
    "online": the loader yields preprocessed images -> ``featurize`` runs the frozen
    VJEPA encoder (4-layer tap) on each batch, so nothing is cached to disk. ``image_hw``
    overrides the input resolution (for curriculum stages); depth stays at ``target_hw``.
    """
    if cfg.feature_mode == "online":
        h, w = image_hw or cfg.image_hw or cfg.target_hw
        common = dict(
            kitti_sequences_dir=cfg.kitti_sequences_dir,
            image_dirname=cfg.image_dirname,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            target_hw=cfg.target_hw,
            image_height=h,
            image_width=w,
        )
        train_ds = ImageDepthDataset(cfg.train_sequences, augment=cfg.augment, **common)
        val_ds = ImageDepthDataset(cfg.val_sequences, augment=False, **common)

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
    align = getattr(cfg, "eval_scale_align", False)
    totals = {n: 0.0 for n in names}
    if align:
        totals["abs_rel_al"] = 0.0
        totals["a1_al"] = 0.0
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
        if align:
            # Median scaling: remove the global scale ambiguity SigLoss leaves, so the
            # aligned metrics reflect predicted *structure* rather than absolute scale.
            vm = valid & (depth > 0)
            scale = (torch.median(depth[vm]) / torch.median(pred[vm]).clamp_min(1e-6)) if vm.any() else pred.new_tensor(1.0)
            pred_al = (pred * scale).clamp(cfg.min_depth, cfg.max_depth)
            ma = calculate_depth_metrics(depth, pred_al, valid)
            totals["abs_rel_al"] += float(ma.abs_rel)
            totals["a1_al"] += float(ma.a1)
        count += 1
    probe.train()
    return {k: v / max(count, 1) for k, v in totals.items()}


def train(cfg: DepthProbeConfig | None = None):
    cfg = cfg or DepthProbeConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Resolve the curriculum: an explicit list of stages, or one stage from the flat fields.
    stages = cfg.stages or [{
        "image_hw": cfg.image_hw or cfg.target_hw, "iters": cfg.total_iters,
        "lr": cfg.lr, "warmup_iters": cfg.warmup_iters,
    }]

    probe = DepthProbe.from_config(cfg).to(device)
    criterion = build_loss(cfg.losses).to(device)
    # One optimiser carried across stages (Adam moments transfer into the fine-tune phase).
    optim = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    accum = max(1, cfg.grad_accum_steps)

    n_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"[train] device={device}  head={cfg.head_type}  feature_mode={cfg.feature_mode}  "
          f"trainable params={n_params:,}  stages={len(stages)}  "
          f"accum={accum}  eff_batch={cfg.batch_size * accum}  augment={cfg.augment}")

    best_abs_rel = float("inf")
    best_ckpt = os.path.join(cfg.out_dir, "depth_probe_best.pt")

    for si, stage in enumerate(stages, start=1):
        image_hw = tuple(stage["image_hw"])
        n_iters = int(stage["iters"])
        peak_lr = float(stage.get("lr", cfg.lr))
        warmup_iters = int(stage.get("warmup_iters", cfg.warmup_iters))
        total_updates = max(1, n_iters // accum)
        warmup_updates = max(1, warmup_iters // accum)
        is_final = si == len(stages)

        # Rebuild loaders + frozen encoder at this stage's input resolution (depth stays
        # at target_hw). The fully-conv DPT head carries over unchanged across grids.
        train_loader, val_loader, featurize = build_loaders(cfg, device, image_hw=image_hw)
        print(f"[stage {si}/{len(stages)}] image_hw={image_hw}  iters={n_iters}  "
              f"peak_lr={peak_lr:.1e}  final={is_final}")

        probe.train()
        optim.zero_grad(set_to_none=True)
        upd = 0
        for it, (x, depth, valid) in enumerate(islice(_infinite(train_loader), n_iters), start=1):
            feat = featurize(x)
            depth, valid = depth.to(device), valid.to(device)
            pred = predict_depth(probe, feat, depth.shape[-2:])
            loss = criterion(pred, depth, valid)
            # AdaBins Chamfer bin-density term on the per-image centers from this forward.
            if getattr(cfg, "adaptive_bins", False) and cfg.bin_chamfer_weight > 0 and probe.bin_centers is not None:
                loss = loss + cfg.bin_chamfer_weight * chamfer_bin_loss(
                    probe.bin_centers, depth, valid, max_depth=cfg.max_depth
                )
            loss = loss / accum
            loss.backward()

            if it % accum == 0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
                for g in optim.param_groups:
                    g["lr"] = _stage_lr(peak_lr, warmup_updates, total_updates, upd)
                optim.step()
                optim.zero_grad(set_to_none=True)
                upd += 1

            if it % cfg.log_every == 0:
                print(f"[train] iter {it:>6}/{n_iters} (stage {si})  loss {loss.item() * accum:.4f}  "
                      f"lr {optim.param_groups[0]['lr']:.2e}")

            if it % cfg.eval_every == 0 or it == n_iters:
                # Sample val during training; full val only at the very end of the run.
                max_b = None if (it == n_iters and is_final) else cfg.eval_max_batches
                metrics = evaluate(probe, val_loader, cfg, device, featurize, max_batches=max_b)
                msg = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[eval ] iter {it:>6} (stage {si})  {msg}")
                # Only track "best" in the final (deployment-resolution) stage, so a coarse
                # low-res checkpoint can't masquerade as best.
                if is_final and metrics["abs_rel"] < best_abs_rel:
                    best_abs_rel = metrics["abs_rel"]
                    torch.save({"head": probe.state_dict(), "cfg": cfg, "metrics": metrics,
                                "stage": si, "image_hw": image_hw, "iter": it}, best_ckpt)
                    print(f"[eval ] new best abs_rel={best_abs_rel:.4f} -> saved {best_ckpt}")

        # Flush leftover accumulated gradients, then snapshot the end of this stage.
        if n_iters % accum != 0:
            torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)
        stage_ckpt = os.path.join(cfg.out_dir, f"depth_probe_stage{si}.pt")
        torch.save({"head": probe.state_dict(), "cfg": cfg, "stage": si, "image_hw": image_hw}, stage_ckpt)
        print(f"[stage {si}/{len(stages)}] done -> {stage_ckpt}")

    print(f"[train] done. best abs_rel={best_abs_rel:.4f}")
    return probe


if __name__ == "__main__":
    train()
