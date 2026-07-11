"""Train the SimpleFeaturePyramid dense probe (task = depth | semseg).

One iteration-based loop over frozen online V-JEPA 2.1 final-layer features. The task registry
(``tasks.py``) supplies the dataset, model (SFP trunk + task output), loss, evaluator, and
prediction images; everything else here is task-agnostic. Metrics + images stream to W&B.

Run:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -m jepa_drive_wm.probes.dense.train --task depth
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from itertools import islice

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DenseConfig
from .features import build_wrapper, make_featurizer
from .io import dense_collate
from .tasks import get_task


def _infinite(loader):
    while True:
        yield from loader


def _lr_at(peak_lr, warmup_updates, total_updates, u, min_ratio=0.01):
    if u < warmup_updates:
        return peak_lr * (u + 1) / max(warmup_updates, 1)
    progress = (u - warmup_updates) / max(total_updates - warmup_updates, 1)
    return peak_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def init_wandb(cfg: DenseConfig):
    if not cfg.wandb:
        return None
    try:
        import wandb
        return wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_entity,
            name=cfg.wandb_run_name or f"{cfg.task}-{cfg.feat_mode}", group=cfg.wandb_group,
            tags=[cfg.task, cfg.feat_mode], config=asdict(cfg),
        )
    except Exception as e:
        print(f"[wandb] disabled ({type(e).__name__}: {e})")
        return None


def build_loaders(cfg, image_hw, task):
    train_ds = task.build_dataset(cfg, cfg.train_sequences, augment=cfg.augment)
    val_ds = task.build_dataset(cfg, cfg.val_sequences, augment=False)
    wrapper = build_wrapper(cfg, image_hw)
    featurize = make_featurizer(wrapper, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              collate_fn=dense_collate, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            collate_fn=dense_collate, pin_memory=True)
    return train_loader, val_loader, featurize


def build_test_loader(cfg, image_hw, task):
    test_ds = task.build_dataset(cfg, cfg.test_sequences, augment=False)
    return DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                      collate_fn=dense_collate, pin_memory=True)


def predict_dense(model, feat, out_hw):
    """Run the model (bf16 autocast on cuda) and bilinearly upsample to GT resolution."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=feat.is_cuda):
        out = model(feat)
    return F.interpolate(out.float(), size=out_hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, loader, cfg, device, featurize, task, max_batches=None) -> dict:
    model.eval()
    ev = task.make_evaluator(cfg)
    batches = loader if max_batches is None else islice(loader, max_batches)
    for x, target, valid in batches:
        feat = featurize(x)
        target, valid = target.to(device), valid.to(device)
        pred = predict_dense(model, feat, target.shape[-2:])
        ev.update(pred, target, valid)
    model.train()
    return ev.result()


def _build_viz(cfg, sequences, image_hw, run, task):
    if run is None or cfg.wandb_num_images <= 0:
        return None, None
    ds = task.build_dataset(cfg, sequences, augment=False)
    n = min(cfg.wandb_num_images, len(ds))
    idxs = [round(k * (len(ds) - 1) / max(n - 1, 1)) for k in range(n)]
    return ds, idxs


@torch.no_grad()
def _log_pred_images(run, model, cfg, featurize, device, ds, idxs, task, split, step, out_dir):
    if run is None or not idxs:
        return
    import wandb
    tmp_dir = os.path.join(out_dir, "wandb_preds")
    was_training = model.training
    model.eval()
    images = []
    for i in idxs:
        x, target, valid = ds[i]
        feat = featurize(x[None].to(device))
        pred = predict_dense(model, feat, target.shape[-2:])
        save_path = os.path.join(tmp_dir, f"{split}_idx{i:05d}.png")
        task.render(cfg, pred, target[None], valid[None], ds, i, save_path)
        images.append(wandb.Image(save_path, caption=f"{split} idx {i}"))
    run.log({f"{split}/preds": images}, step=step)
    if was_training:
        model.train()


def train(cfg: DenseConfig | None = None):
    cfg = cfg or DenseConfig()
    task = get_task(cfg.task)
    if cfg.feat_mode not in ("final", "pred"):
        raise ValueError(f"feat_mode must be 'final' or 'pred', got {cfg.feat_mode!r}")
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg.out_dir, f"{cfg.task}_{cfg.feat_mode}")
    os.makedirs(out_dir, exist_ok=True)
    run = init_wandb(cfg)

    image_hw = tuple(cfg.image_hw or cfg.target_hw)
    n_iters = cfg.total_iters
    accum = max(1, cfg.grad_accum_steps)
    total_updates = max(1, n_iters // accum)
    warmup_updates = max(1, cfg.warmup_iters // accum)

    model = task.build_model(cfg).to(device)
    criterion = task.build_criterion(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sel_name, sel_lower = task.selection

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] device={device}  task={cfg.task}  feat={cfg.feat_mode}  head=sfp  "
          f"trainable params={n_params:,}  accum={accum}  eff_batch={cfg.batch_size * accum}  "
          f"augment={cfg.augment}  out={out_dir}")

    train_loader, val_loader, featurize = build_loaders(cfg, image_hw, task)
    val_viz_ds, val_viz_idxs = _build_viz(cfg, cfg.val_sequences, image_hw, run, task)

    best = float("inf") if sel_lower else float("-inf")
    best_ckpt = os.path.join(out_dir, "probe_best.pt")

    model.train()
    optim.zero_grad(set_to_none=True)
    upd = 0
    for it, (x, target, valid) in enumerate(islice(_infinite(train_loader), n_iters), start=1):
        feat = featurize(x)
        target, valid = target.to(device), valid.to(device)
        pred = predict_dense(model, feat, target.shape[-2:])
        loss = criterion(pred, target, valid)
        (loss / accum).backward()

        if it % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for g in optim.param_groups:
                g["lr"] = _lr_at(cfg.lr, warmup_updates, total_updates, upd)
            optim.step()
            optim.zero_grad(set_to_none=True)
            upd += 1

        if it % cfg.log_every == 0:
            cur_lr = optim.param_groups[0]["lr"]
            print(f"[train] iter {it:>6}/{n_iters}  loss {loss.item():.4f}  lr {cur_lr:.2e}")
            if run is not None:
                run.log({"train/loss": loss.item(), "train/lr": cur_lr}, step=it)

        if it % cfg.eval_every == 0 or it == n_iters:
            max_b = None if it == n_iters else cfg.eval_max_batches
            metrics = evaluate(model, val_loader, cfg, device, featurize, task, max_batches=max_b)
            print(f"[eval ] iter {it:>6}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if run is not None:
                run.log({f"val/{k}": v for k, v in metrics.items()}, step=it)
                _log_pred_images(run, model, cfg, featurize, device, val_viz_ds, val_viz_idxs or [],
                                 task, "val", it, out_dir)
            improved = metrics[sel_name] < best if sel_lower else metrics[sel_name] > best
            if improved:
                best = metrics[sel_name]
                torch.save({"model": model.state_dict(), "cfg": cfg, "metrics": metrics, "iter": it,
                            "image_hw": image_hw}, best_ckpt)
                print(f"[eval ] new best {sel_name}={best:.4f} -> saved {best_ckpt}")

    if n_iters % accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)
    stage_ckpt = os.path.join(out_dir, "probe_final.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg, "image_hw": image_hw}, stage_ckpt)
    print(f"[train] done -> {stage_ckpt}")

    # Final held-out TEST eval on the final-iter weights (never used for selection).
    test_loader = build_test_loader(cfg, image_hw, task)
    test_metrics = evaluate(model, test_loader, cfg, device, featurize, task, max_batches=None)
    print(f"[test ] seqs={list(cfg.test_sequences)} (final-iter)  "
          + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
    if run is not None:
        run.log({f"test/{k}": v for k, v in test_metrics.items()}, step=n_iters)
        run.summary.update({f"test/{k}": v for k, v in test_metrics.items()})
        test_viz_ds, test_viz_idxs = _build_viz(cfg, cfg.test_sequences, image_hw, run, task)
        _log_pred_images(run, model, cfg, featurize, device, test_viz_ds, test_viz_idxs or [],
                         task, "test", n_iters, out_dir)
        run.finish()

    print(f"[train] done. best(val) {sel_name}={best:.4f}  test {sel_name}={test_metrics[sel_name]:.4f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the SFP dense probe (depth | semseg).")
    p.add_argument("--task", choices=("depth", "semseg"), default="depth")
    p.add_argument("--feat", choices=("final", "pred"), default="final", help="final (768-d) or pred (384-d)")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = p.parse_args()
    train(DenseConfig(task=args.task, feat_mode=args.feat, wandb=args.wandb))
