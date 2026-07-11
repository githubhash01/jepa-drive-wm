"""Task registry: the thin per-task bundle over the shared SFP architecture.

Each task supplies its dataset (ground truth), the full model (SFP trunk + task output),
loss, evaluator (metric accumulation), the model-selection metric, and prediction images.
Everything else (featurizer, trunk, trainer) is shared.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from . import taxonomy
from .binning import FeaturesToDepth
from .io import DepthDataset, SemSegDataset
from .losses import build_loss
from .metrics import calculate_depth_metrics, intersect_and_union
from .sfp import SimpleFeaturePyramid
from .viz import save_depth_triptych, save_semseg_triptych


# ----------------------------------------------------------------------------- Depth
class _DepthEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.names = ("a1", "a2", "a3", "abs_rel", "rmse", "silog")
        self.totals = {n: 0.0 for n in self.names}
        self.align = cfg.eval_scale_align
        if self.align:
            self.totals["abs_rel_al"] = 0.0
            self.totals["a1_al"] = 0.0
        self.count = 0

    def update(self, pred, target, valid):
        cfg = self.cfg
        pred = pred.clamp(cfg.min_depth, cfg.max_depth)
        m = calculate_depth_metrics(target, pred, valid)
        for n in self.names:
            self.totals[n] += float(getattr(m, n))
        if self.align:
            vm = valid & (target > 0)
            scale = (torch.median(target[vm]) / torch.median(pred[vm]).clamp_min(1e-6)) if vm.any() else pred.new_tensor(1.0)
            ma = calculate_depth_metrics(target, (pred * scale).clamp(cfg.min_depth, cfg.max_depth), valid)
            self.totals["abs_rel_al"] += float(ma.abs_rel)
            self.totals["a1_al"] += float(ma.a1)
        self.count += 1

    def result(self):
        return {k: v / max(self.count, 1) for k, v in self.totals.items()}


class DepthTask:
    name = "depth"
    selection = ("abs_rel", True)   # (metric, lower_is_better)

    def build_dataset(self, cfg, sequences, augment):
        return DepthDataset(
            sequences, cfg.kitti_sequences_dir, cfg.image_dirname, cfg.target_hw,
            cfg.image_hw[0] if cfg.image_hw else cfg.target_hw[0],
            cfg.image_hw[1] if cfg.image_hw else cfg.target_hw[1],
            augment, min_depth=cfg.min_depth, max_depth=cfg.max_depth,
        )

    def build_model(self, cfg):
        return nn.Sequential(
            SimpleFeaturePyramid.from_config(cfg),
            FeaturesToDepth(min_depth=cfg.min_depth, max_depth=cfg.max_depth, bins_strategy=cfg.bins_strategy,
                            norm_strategy=cfg.norm_strategy, bin_min_depth=cfg.bin_min_depth,
                            bin_max_depth=cfg.bin_max_depth),
        )

    def build_criterion(self, cfg):
        return build_loss(cfg.depth_losses)

    def make_evaluator(self, cfg):
        return _DepthEvaluator(cfg)

    @torch.no_grad()
    def render(self, cfg, pred, target, valid, ds, idx, save_path):
        from PIL import Image
        depth = pred.clamp(cfg.min_depth, cfg.max_depth)[0, 0].cpu().numpy()
        gt_vis = np.where(valid[0, 0].cpu().numpy() > 0, target[0, 0].cpu().numpy(), np.nan)
        img_path = ds.samples[idx][0]
        rgb = np.asarray(Image.open(img_path).convert("RGB")) if os.path.exists(img_path) else None
        save_depth_triptych(rgb, gt_vis, depth, save_path,
                            pred_title="Predicted depth (V-JEPA 2.1 SFP probe)",
                            rgb_name=os.path.basename(img_path))


# ----------------------------------------------------------------------------- SemSeg
class _SegEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        C, G = cfg.num_classes, taxonomy.NUM_GROUPS
        self.inter = torch.zeros(C, dtype=torch.long)
        self.union = torch.zeros(C, dtype=torch.long)
        self.g_inter = torch.zeros(G, dtype=torch.long)
        self.g_union = torch.zeros(G, dtype=torch.long)
        self._c2g = torch.from_numpy(taxonomy.CLASS_TO_GROUP)

    def update(self, pred, target, valid):
        cfg = self.cfg
        pred_lab = pred.argmax(dim=1).cpu()             # (B, H, W)
        tgt = target[:, 0].long().cpu()                 # (B, H, W)
        if valid is not None:
            tgt = tgt.masked_fill(~valid[:, 0].bool().cpu(), cfg.ignore_index)
        i, u = intersect_and_union(pred_lab, tgt, cfg.num_classes, cfg.ignore_index)
        self.inter += i; self.union += u
        # coarse groups
        c2g = self._c2g
        pred_g = c2g[pred_lab]
        tgt_g = torch.where(tgt == cfg.ignore_index, torch.full_like(tgt, cfg.ignore_index), c2g[tgt.clamp(min=0)])
        gi, gu = intersect_and_union(pred_g, tgt_g, taxonomy.NUM_GROUPS, cfg.ignore_index)
        self.g_inter += gi; self.g_union += gu

    def result(self):
        iou = self.inter.float() / self.union.float().clamp_min(1)
        giou = self.g_inter.float() / self.g_union.float().clamp_min(1)
        present = self.union > 0
        g_present = self.g_union > 0
        return {
            "miou": float(iou[present].mean()) if present.any() else 0.0,
            "group_miou": float(giou[g_present].mean()) if g_present.any() else 0.0,
            "pix_acc": float(self.inter.sum() / self.union.sum().clamp_min(1)),
        }


class SemSegTask:
    name = "semseg"
    selection = ("miou", False)

    def build_dataset(self, cfg, sequences, augment):
        return SemSegDataset(
            sequences, cfg.kitti_sequences_dir, cfg.image_dirname, cfg.target_hw,
            cfg.image_hw[0] if cfg.image_hw else cfg.target_hw[0],
            cfg.image_hw[1] if cfg.image_hw else cfg.target_hw[1],
            augment, semantic_root=cfg.semantic_root,
        )

    def build_model(self, cfg):
        return SimpleFeaturePyramid.from_config(cfg)   # logits, no depth decode

    def build_criterion(self, cfg):
        return build_loss(cfg.semseg_losses, loss_kwargs={"CE": {"ignore_index": cfg.ignore_index}})

    def make_evaluator(self, cfg):
        return _SegEvaluator(cfg)

    @torch.no_grad()
    def render(self, cfg, pred, target, valid, ds, idx, save_path):
        from PIL import Image
        pred_lab = pred.argmax(dim=1)[0].cpu().numpy()
        gt_lab = target[0, 0].cpu().numpy()
        img_path = ds.samples[idx][0]
        rgb = np.asarray(Image.open(img_path).convert("RGB")) if os.path.exists(img_path) else None
        save_semseg_triptych(rgb, gt_lab, pred_lab, taxonomy.PALETTE, save_path,
                             pred_title="Predicted segmentation (V-JEPA 2.1 SFP probe)",
                             class_names=taxonomy.CLASS_NAMES, rgb_name=os.path.basename(img_path))


TASKS = {"depth": DepthTask(), "semseg": SemSegTask()}


def get_task(name: str):
    if name not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}, got {name!r}")
    return TASKS[name]
