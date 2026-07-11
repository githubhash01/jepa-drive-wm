"""Train the 4-layer DPT depth probe with online feature extraction.

Iteration-based loop (no Lightning). The V-JEPA encoder is frozen but runs *in the loop*:
each batch of images is encoded to the 4-layer tap on the fly (no disk cache -- the 4-layer
cache is too big). Only the DPT head trains. Predicted depth is upsampled to the GT
resolution before loss/metrics. Metrics + prediction images stream to Weights & Biases
(on by default; ``--no-wandb`` or a missing login disables it gracefully).

Optional curriculum (``cfg.stages``): re-encode at a lower resolution first, then fine-tune
at full res -- the fully-conv DPT head carries over. One AdamW optimiser spans all stages.

Run:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -m jepa_drive_wm.probes.depth.dpt_probe.train
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict
from itertools import islice

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .._core import build_loss, calculate_depth_metrics, depth_collate
from .config import DPTProbeConfig
from .dataset import ImageDepthDataset
from .head import build_probe


def _infinite(loader):
    while True:
        yield from loader


def init_wandb(cfg: DPTProbeConfig):
    """Start a W&B run, or return ``None`` if disabled / login missing (never blocks a run)."""
    if not cfg.wandb:
        return None
    try:
        import wandb
        return wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_entity,
            name=cfg.wandb_run_name or f"{cfg.head_type}-{cfg.layer_mode}", group=cfg.wandb_group,
            tags=[cfg.head_type, cfg.layer_mode], config=asdict(cfg),
        )
    except Exception as e:  # not logged in, offline failure, etc. -> carry on without it.
        print(f"[wandb] disabled ({type(e).__name__}: {e})")
        return None


def _build_viz(cfg, sequences, image_hw, run):
    """Build a preview dataset + spread-out indices for W&B images (or (None, None))."""
    if run is None or cfg.wandb_num_images <= 0:
        return None, None
    ds = ImageDepthDataset(
        sequences, augment=False,
        kitti_sequences_dir=cfg.kitti_sequences_dir, image_dirname=cfg.image_dirname,
        min_depth=cfg.min_depth, max_depth=cfg.max_depth, target_hw=cfg.target_hw,
        image_height=image_hw[0], image_width=image_hw[1],
    )
    n = min(cfg.wandb_num_images, len(ds))
    idxs = [round(k * (len(ds) - 1) / max(n - 1, 1)) for k in range(n)]
    return ds, idxs


@torch.no_grad()
def _log_pred_images(run, probe, cfg, featurize, device, ds, idxs, split, step, out_dir):
    """Log RGB|GT|pred depth triptychs for a few fixed test frames as ``wandb.Image``."""
    if run is None or cfg.wandb_num_images <= 0:
        return
    import numpy as np
    import wandb
    from PIL import Image
    from .._core.viz import image_path_from_depth, save_depth_triptych

    tmp_dir = os.path.join(out_dir, "wandb_preds")
    os.makedirs(tmp_dir, exist_ok=True)
    label = {"quad": "4-layer", "final": "final-layer", "pred": "predictor-384"}[cfg.layer_mode]
    was_training = probe.training
    probe.eval()
    images = []
    for i in idxs:
        x, depth, valid = ds[i]
        depth_path = ds.samples[i][1]
        feat = featurize(x[None].to(device))
        pred = predict_depth(probe, feat, depth.shape[-2:]).clamp(cfg.min_depth, cfg.max_depth)
        pred = pred[0, 0].cpu().numpy()
        gt_vis = np.where(valid[0].numpy(), depth[0].numpy(), np.nan)
        img_path = image_path_from_depth(depth_path, cfg.image_dirname)
        rgb = np.asarray(Image.open(img_path).convert("RGB")) if os.path.exists(img_path) else None
        save_path = os.path.join(tmp_dir, f"{split}_idx{i:05d}.png")
        save_depth_triptych(rgb, gt_vis, pred, save_path,
                            pred_title=f"Predicted depth (V-JEPA 2.1 {label} DPT probe)",
                            rgb_name=os.path.basename(img_path))
        images.append(wandb.Image(save_path, caption=f"{split} idx {i}"))
    run.log({f"{split}/preds": images}, step=step)
    if was_training:
        probe.train()


def _lr_at(peak_lr, warmup_updates, total_updates, u, min_ratio=0.01):
    """Linear warm-up to ``peak_lr``, then cosine decay to ``peak_lr * min_ratio``."""
    if u < warmup_updates:
        return peak_lr * (u + 1) / max(warmup_updates, 1)
    progress = (u - warmup_updates) / max(total_updates - warmup_updates, 1)
    return peak_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_loaders(cfg: DPTProbeConfig, image_hw):
    """Build train/val loaders + a ``featurize`` that encodes images to the 4-layer tap."""
    h, w = image_hw
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
    wrapper = VJEPA21Wrapper(size=VJEPA21Size[cfg.vjepa_size], image_height=h, image_width=w, verbose=True)

    def featurize(x: torch.Tensor) -> torch.Tensor:
        # One encoder pass yields all 4 tapped layers [L3,L6,L9,L12]. The last tap is the
        # post-final-norm final grid (matches encode_images). The DPT head consumes a 4-layer
        # stack; the linear head consumes the single final grid.
        feat = wrapper.extract_hierarchical(x)   # (B, 4, D=768, gh, gw)
        if cfg.head_type == "dpt" and cfg.layer_mode == "quad":
            return feat
        final = feat[:, -1]                                  # (B, 768, gh, gw) = L12 (final-normed)
        if cfg.layer_mode == "pred":
            final = wrapper.compress_final(final)            # (B, 384, gh, gw) = predictor_embed(L12)
        if cfg.head_type == "linear":
            return final                                     # (B, D', gh, gw) single grid
        return final.unsqueeze(1).repeat(1, 4, 1, 1, 1)      # DPT final/pred: (B, 4, D', gh, gw)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=depth_collate, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=depth_collate, pin_memory=True,
    )
    return train_loader, val_loader, featurize


def build_test_loader(cfg: DPTProbeConfig, image_hw):
    """Loader over the held-out ``test_sequences`` (final reporting only, never selection)."""
    h, w = image_hw
    test_ds = ImageDepthDataset(
        cfg.test_sequences, augment=False,
        kitti_sequences_dir=cfg.kitti_sequences_dir, image_dirname=cfg.image_dirname,
        min_depth=cfg.min_depth, max_depth=cfg.max_depth, target_hw=cfg.target_hw,
        image_height=h, image_width=w,
    )
    return DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=depth_collate, pin_memory=True,
    )


def predict_depth(probe, feat, out_hw):
    """Run the probe (bf16 autocast on cuda) and upsample depth to GT resolution (B,1,H,W)."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=feat.is_cuda):
        depth = probe(feat)
    return F.interpolate(depth.float(), size=out_hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(probe, val_loader, cfg, device, featurize, max_batches=None) -> dict:
    probe.eval()
    names = ("a1", "a2", "a3", "abs_rel", "rmse", "silog")
    align = cfg.eval_scale_align
    totals = {n: 0.0 for n in names}
    if align:
        totals["abs_rel_al"] = 0.0
        totals["a1_al"] = 0.0
    count = 0
    batches = val_loader if max_batches is None else islice(val_loader, max_batches)
    for x, depth, valid in batches:
        feat = featurize(x)
        depth, valid = depth.to(device), valid.to(device)
        pred = predict_depth(probe, feat, depth.shape[-2:]).clamp(cfg.min_depth, cfg.max_depth)
        m = calculate_depth_metrics(depth, pred, valid)
        for n in names:
            totals[n] += float(getattr(m, n))
        if align:
            vm = valid & (depth > 0)
            scale = (torch.median(depth[vm]) / torch.median(pred[vm]).clamp_min(1e-6)) if vm.any() else pred.new_tensor(1.0)
            pred_al = (pred * scale).clamp(cfg.min_depth, cfg.max_depth)
            ma = calculate_depth_metrics(depth, pred_al, valid)
            totals["abs_rel_al"] += float(ma.abs_rel)
            totals["a1_al"] += float(ma.a1)
        count += 1
    probe.train()
    return {k: v / max(count, 1) for k, v in totals.items()}


def train(cfg: DPTProbeConfig | None = None):
    cfg = cfg or DPTProbeConfig()
    if cfg.layer_mode not in ("quad", "final", "pred"):
        raise ValueError(f"layer_mode must be 'quad', 'final' or 'pred', got {cfg.layer_mode!r}")
    if cfg.head_type not in ("dpt", "linear"):
        raise ValueError(f"head_type must be 'dpt' or 'linear', got {cfg.head_type!r}")
    if cfg.head_type == "linear" and cfg.layer_mode == "quad":
        raise ValueError("head_type='linear' is single-layer; use layer_mode 'final' or 'pred', not 'quad'.")
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Per-(head, mode) output dir so experiment arms never clobber each other.
    out_dir = os.path.join(cfg.out_dir, f"{cfg.head_type}_{cfg.layer_mode}")
    os.makedirs(out_dir, exist_ok=True)
    run = init_wandb(cfg)

    # Resolve the curriculum: an explicit list of stages, or one stage from the flat fields.
    stages = cfg.stages or [{
        "image_hw": cfg.image_hw or cfg.target_hw, "iters": cfg.total_iters,
        "lr": cfg.lr, "warmup_iters": cfg.warmup_iters,
    }]

    probe = build_probe(cfg).to(device)
    criterion = build_loss(cfg.losses).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    accum = max(1, cfg.grad_accum_steps)

    n_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"[train] device={device}  head={cfg.head_type}  layer_mode={cfg.layer_mode}  "
          f"trainable params={n_params:,}  stages={len(stages)}  "
          f"accum={accum}  eff_batch={cfg.batch_size * accum}  augment={cfg.augment}  out={out_dir}")

    best_abs_rel = float("inf")
    best_ckpt = os.path.join(out_dir, "depth_probe_best.pt")
    seen = 0  # cumulative frames across stages -> global x-axis for W&B

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
        train_loader, val_loader, featurize = build_loaders(cfg, image_hw=image_hw)
        print(f"[stage {si}/{len(stages)}] image_hw={image_hw}  iters={n_iters}  "
              f"peak_lr={peak_lr:.1e}  final={is_final}")

        # Fixed VAL frames for the during-training W&B image panel (test stays held out
        # visually until the final [test] eval below).
        val_viz_ds, val_viz_idxs = _build_viz(cfg, cfg.val_sequences, image_hw, run)

        probe.train()
        optim.zero_grad(set_to_none=True)
        upd = 0
        for it, (x, depth, valid) in enumerate(islice(_infinite(train_loader), n_iters), start=1):
            feat = featurize(x)
            depth, valid = depth.to(device), valid.to(device)
            pred = predict_depth(probe, feat, depth.shape[-2:])
            loss = criterion(pred, depth, valid)
            (loss / accum).backward()

            if it % accum == 0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
                for g in optim.param_groups:
                    g["lr"] = _lr_at(peak_lr, warmup_updates, total_updates, upd)
                optim.step()
                optim.zero_grad(set_to_none=True)
                upd += 1

            if it % cfg.log_every == 0:
                cur_lr = optim.param_groups[0]["lr"]
                print(f"[train] iter {it:>6}/{n_iters} (stage {si})  loss {loss.item():.4f}  "
                      f"lr {cur_lr:.2e}")
                if run is not None:
                    run.log({"train/loss": loss.item(), "train/lr": cur_lr, "stage": si}, step=seen + it)

            if it % cfg.eval_every == 0 or it == n_iters:
                max_b = None if (it == n_iters and is_final) else cfg.eval_max_batches
                metrics = evaluate(probe, val_loader, cfg, device, featurize, max_batches=max_b)
                print(f"[eval ] iter {it:>6} (stage {si})  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                if run is not None:
                    run.log({f"val/{k}": v for k, v in metrics.items()}, step=seen + it)
                    _log_pred_images(run, probe, cfg, featurize, device, val_viz_ds, val_viz_idxs or [],
                                     split="val", step=seen + it, out_dir=out_dir)
                # Only track "best" in the final (deployment-resolution) stage.
                if is_final and metrics["abs_rel"] < best_abs_rel:
                    best_abs_rel = metrics["abs_rel"]
                    torch.save({"head": probe.state_dict(), "cfg": cfg, "metrics": metrics,
                                "stage": si, "image_hw": image_hw, "iter": it}, best_ckpt)
                    print(f"[eval ] new best abs_rel={best_abs_rel:.4f} -> saved {best_ckpt}")

        if n_iters % accum != 0:
            torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)
        stage_ckpt = os.path.join(out_dir, f"depth_probe_stage{si}.pt")
        torch.save({"head": probe.state_dict(), "cfg": cfg, "stage": si, "image_hw": image_hw}, stage_ckpt)
        print(f"[stage {si}/{len(stages)}] done -> {stage_ckpt}")
        seen += n_iters

    # Final held-out TEST eval on the final-iter weights (full test set, never used for
    # selection). This is the unbiased number to report for the research comparison.
    test_loader = build_test_loader(cfg, image_hw=image_hw)
    test_metrics = evaluate(probe, test_loader, cfg, device, featurize, max_batches=None)
    print(f"[test ] seqs={list(cfg.test_sequences)} (final-iter weights)  "
          + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
    if run is not None:
        run.log({f"test/{k}": v for k, v in test_metrics.items()}, step=seen)
        run.summary.update({f"test/{k}": v for k, v in test_metrics.items()})
        test_viz_ds, test_viz_idxs = _build_viz(cfg, cfg.test_sequences, image_hw, run)
        _log_pred_images(run, probe, cfg, featurize, device, test_viz_ds, test_viz_idxs or [],
                         split="test", step=seen, out_dir=out_dir)
        run.finish()

    print(f"[train] done. best(val) abs_rel={best_abs_rel:.4f}  test abs_rel={test_metrics['abs_rel']:.4f}")
    return probe


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train the DPT depth probe (final-vs-4-layer experiment).")
    p.add_argument("--mode", choices=("quad", "final", "pred"), default="quad",
                   help="quad = 4 layers [L3,L6,L9,L12]; final = L12 x4 (768-d); "
                        "pred = predictor_embed(L12) x4 (384-d compression)")
    p.add_argument("--head", choices=("dpt", "linear"), default="dpt",
                   help="dpt = 4-layer DPT decoder; linear = DINOv3-style dense linear probe "
                        "(final/pred only)")
    p.add_argument("--no-wandb", dest="wandb", action="store_false",
                   help="disable Weights & Biases logging (on by default)")
    args = p.parse_args()
    train(DPTProbeConfig(layer_mode=args.mode, head_type=args.head, wandb=args.wandb))
