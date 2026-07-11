"""Metrics: monocular-depth (a1/a2/a3, abs_rel, rmse, silog, ...) and segmentation IoU."""
from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _DepthMetric:
    name: str
    is_lower_better: bool


DEPTH_METRICS = (
    _DepthMetric("a1", False), _DepthMetric("a2", False), _DepthMetric("a3", False),
    _DepthMetric("abs_rel", True), _DepthMetric("rmse", True), _DepthMetric("log_10", True),
    _DepthMetric("rmse_log", True), _DepthMetric("silog", True), _DepthMetric("sq_rel", True),
    _DepthMetric("mae", True),
)
_DepthMetricValues = namedtuple("DepthMetricValues", [m.name for m in DEPTH_METRICS])


def calculate_depth_metrics(gt, pred, valid_mask=None):
    if gt.shape[0] == 0:
        return _DepthMetricValues(**{m.name: torch.nan for m in DEPTH_METRICS})
    if valid_mask is not None:
        valid_mask = torch.logical_and(valid_mask, gt > 0)
    gt = gt[valid_mask]
    pred = pred[valid_mask]

    md = {}
    thresh = torch.maximum((gt / pred), (pred / gt))
    md["a1"] = (thresh < 1.25).float().mean()
    md["a2"] = (thresh < 1.25 ** 2).float().mean()
    md["a3"] = (thresh < 1.25 ** 3).float().mean()
    error = gt - pred
    sq_error = error ** 2
    md["mae"] = torch.mean(torch.abs(error))
    md["abs_rel"] = torch.mean(torch.abs(error) / gt)
    md["sq_rel"] = torch.mean(sq_error / gt)
    md["rmse"] = torch.sqrt(sq_error.mean())
    error_log = torch.log(gt) - torch.log(pred)
    md["rmse_log"] = torch.sqrt((error_log ** 2).mean())
    silog = torch.sqrt(torch.mean(error_log ** 2) - torch.mean(error_log) ** 2) * 100
    md["silog"] = torch.tensor(0.0) if torch.isnan(silog) else silog
    md["log_10"] = (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()
    return _DepthMetricValues(**md)


def intersect_and_union(pred_labels, target_labels, num_classes, ignore_index=255):
    """Per-class intersection & union areas for a batch of label maps.

    ``pred_labels``/``target_labels``: integer tensors, same spatial shape. Returns two
    ``(num_classes,)`` long tensors to be accumulated across the eval set before dividing.
    """
    pred = pred_labels.flatten()
    target = target_labels.flatten()
    keep = target != ignore_index
    pred, target = pred[keep], target[keep]
    inter = pred[pred == target]
    area_inter = torch.histc(inter.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_pred = torch.histc(pred.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_pred + area_target - area_inter
    return area_inter.long(), area_union.long()
