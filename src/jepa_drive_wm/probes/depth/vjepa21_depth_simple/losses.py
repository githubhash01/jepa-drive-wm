"""
Depth losses, copied from the DINOv3 depth eval (probes/depth/dinov3_depth/loss.py)
so this package stays self-contained and does not import ``dinov3.*``.

SigLoss is the scale-invariant log-depth loss (AdaBins / BinsFormer). MultiLoss lets
you weight several together; the probe defaults to ``{SIGLOSS: 1.0}``.
"""
from __future__ import annotations

from enum import Enum
from functools import partial

import torch
from torch import nn


class LossType(Enum):
    SIGLOSS = "sigloss"
    GRADIENT_LOSS = "gradient_loss"
    GRADIENT_LOG_LOSS = "gradient_log_loss"
    L1 = "l1"

    def module(self, *args, **kwargs):
        return {
            LossType.SIGLOSS: partial(SigLoss, warm_up=True, warm_iter=100),
            LossType.GRADIENT_LOG_LOSS: GradientLogLoss,
            LossType.GRADIENT_LOSS: GradientLoss,
            LossType.L1: L1Loss,
        }[self](*args, **kwargs)


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 0.001

    def forward(self, input, target, valid_mask=None):
        input_downscaled = [input] + [input[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target] + [target[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        if valid_mask is not None:
            mask_downscaled = [valid_mask] + [valid_mask[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        else:
            mask_downscaled = [torch.ones_like(t, dtype=bool) for t in target_downscaled]

        gradient_loss = 0
        for input, target, mask in zip(input_downscaled, target_downscaled, mask_downscaled):
            N = torch.sum(mask)
            d_diff = torch.mul(input - target, mask)

            v_gradient = torch.abs(d_diff[..., 0:-2, :] - d_diff[..., 2:, :])
            v_mask = torch.mul(mask[..., 0:-2, :], mask[..., 2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(d_diff[..., :, 0:-2] - d_diff[..., :, 2:])
            h_mask = torch.mul(mask[..., :, 0:-2], mask[..., :, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)
            gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N

        return gradient_loss


class GradientLogLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 0.001

    def forward(self, input, target, valid_mask=None):
        input_downscaled = [input] + [input[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target] + [target[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        if valid_mask is not None:
            mask_downscaled = [valid_mask] + [valid_mask[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        else:
            mask_downscaled = [torch.ones_like(t, dtype=bool) for t in target_downscaled]

        gradient_loss = 0
        for input, target, mask in zip(input_downscaled, target_downscaled, mask_downscaled):
            N = torch.sum(mask)
            input_log = torch.log(input + self.eps)
            target_log = torch.log(target + self.eps)
            log_d_diff = input_log - target_log
            log_d_diff = torch.mul(log_d_diff, mask)

            v_gradient = torch.abs(log_d_diff[..., 0:-2, :] - log_d_diff[..., 2:, :])
            v_mask = torch.mul(mask[..., 0:-2, :], mask[..., 2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(log_d_diff[..., :, 0:-2] - log_d_diff[..., :, 2:])
            h_mask = torch.mul(mask[..., :, 0:-2], mask[..., :, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)
            gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N

        return gradient_loss


class L1Loss(nn.Module):
    def forward(self, input, target, valid_mask=None):
        loss = nn.functional.l1_loss(input, target, reduction="none")
        mask = valid_mask if (valid_mask is not None) else torch.ones_like(input, dtype=bool)
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-7)


class SigLoss(nn.Module):
    """Scale-invariant log loss (adapted from BinsFormer / AdaBins)."""

    def __init__(self, warm_up=True, warm_iter=100):
        super().__init__()
        self.loss_name = "SigLoss"
        self.eps = 0.001  # avoid grad explode
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones_like(target, dtype=bool)
        input = input[valid_mask]
        target = target[valid_mask]

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = 0.15 * torch.pow(torch.mean(g), 2)
        if self.warm_up and self.warm_up_counter < self.warm_iter:
            self.warm_up_counter += 1
        else:
            Dg += torch.var(g)
        if Dg <= 0:
            return torch.abs(Dg)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt, valid_mask=None):
        return self.sigloss(depth_pred, depth_gt, valid_mask)


class MultiLoss(nn.Module):
    """Weighted sum of losses, keyed by ``LossType``: ``{LossType.SIGLOSS: 1.0, ...}``."""

    def __init__(self, dict_losses: dict[LossType, float]):
        super().__init__()
        self.dict_losses = nn.ModuleDict({lt.name: lt.module() for lt in dict_losses})
        self.dict_weights = {lt.name: w for (lt, w) in dict_losses.items()}

    def forward(self, depth_pred, depth_gt, valid_mask=None):
        loss_depth = 0
        for name in self.dict_losses:
            loss_depth += self.dict_weights[name] * self.dict_losses[name](depth_pred, depth_gt, valid_mask)
        return loss_depth


def build_loss(losses: dict[str, float]) -> MultiLoss:
    """Build a MultiLoss from a ``{"SIGLOSS": 1.0}``-style config dict."""
    return MultiLoss({LossType[name]: weight for name, weight in losses.items()})


def chamfer_bin_loss(
    centers: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    max_depth: float = 80.0,
    n_sample: int = 512,
) -> torch.Tensor:
    """Bi-directional Chamfer distance between predicted bin centers and GT depths (AdaBins).

    Encourages the adaptive bin centers to (a) sit near real depths and (b) collectively
    cover the depth distribution, so they don't collapse. Distances are computed on
    depth/max_depth (in [0,1]) for a scale-stable term. ``centers`` (B, n_bins), ``depth``
    and ``valid`` (B, 1, H, W).
    """
    B = centers.shape[0]
    total = centers.new_zeros(())
    count = 0
    c = centers / max_depth  # (B, n_bins) normalised
    for b in range(B):
        d = depth[b][valid[b]]
        if d.numel() == 0:
            continue
        d = d / max_depth
        if d.numel() > n_sample:
            idx = torch.randint(0, d.numel(), (n_sample,), device=d.device)
            d = d[idx]
        dist = (d[:, None] - c[b][None, :]).abs()      # (M, n_bins)
        gt_to_bin = dist.min(dim=1).values.mean()      # each GT depth -> nearest bin
        bin_to_gt = dist.min(dim=0).values.mean()      # each bin -> nearest GT depth
        total = total + gt_to_bin + bin_to_gt
        count += 1
    return total / max(count, 1)
