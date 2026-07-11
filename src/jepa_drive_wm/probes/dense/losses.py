"""Losses for the dense probe.

Depth: SigLoss (scale-invariant log, primary) + optional multi-scale gradient matching + L1.
Semseg: pixelwise CrossEntropy. ``MultiLoss`` weights any subset; ``build_loss`` builds it from
a ``{"SIGLOSS": 1.0, ...}`` / ``{"CE": 1.0}`` config dict. Every loss takes ``(pred, gt, valid)``.
"""
from __future__ import annotations

from enum import Enum
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


class LossType(Enum):
    SIGLOSS = "sigloss"
    GRADIENT_LOSS = "gradient_loss"
    GRADIENT_LOG_LOSS = "gradient_log_loss"
    L1 = "l1"
    CE = "ce"

    def module(self, *args, **kwargs):
        return {
            LossType.SIGLOSS: partial(SigLoss, warm_up=True, warm_iter=100),
            LossType.GRADIENT_LOG_LOSS: GradientLogLoss,
            LossType.GRADIENT_LOSS: GradientLoss,
            LossType.L1: L1Loss,
            LossType.CE: CrossEntropyLoss,
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
            log_d_diff = torch.mul(input_log - target_log, mask)
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
        loss = F.l1_loss(input, target, reduction="none")
        mask = valid_mask if (valid_mask is not None) else torch.ones_like(input, dtype=bool)
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-7)


class SigLoss(nn.Module):
    """Scale-invariant log loss (Eigen SILog, lambda=0.85; AdaBins/BinsFormer form)."""

    def __init__(self, warm_up=True, warm_iter=100):
        super().__init__()
        self.loss_name = "SigLoss"
        self.eps = 0.001
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


class CrossEntropyLoss(nn.Module):
    """Pixelwise cross-entropy for semantic segmentation.

    ``logits`` (B, C, H, W); ``target`` (B, 1, H, W) or (B, H, W) long class ids; ``valid``
    (same spatial shape) bool -- invalid pixels are set to ``ignore_index`` and skipped.
    """

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target, valid_mask=None):
        if target.ndim == 4:
            target = target[:, 0]
        target = target.long()
        if valid_mask is not None:
            vm = valid_mask[:, 0] if valid_mask.ndim == 4 else valid_mask
            target = target.masked_fill(~vm.bool(), self.ignore_index)
        return F.cross_entropy(logits, target, ignore_index=self.ignore_index)


class MultiLoss(nn.Module):
    """Weighted sum of losses keyed by ``LossType``: ``{LossType.SIGLOSS: 1.0, ...}``."""

    def __init__(self, dict_losses: dict, loss_kwargs: dict | None = None):
        super().__init__()
        loss_kwargs = loss_kwargs or {}
        self.dict_losses = nn.ModuleDict({lt.name: lt.module(**loss_kwargs.get(lt.name, {})) for lt in dict_losses})
        self.dict_weights = {lt.name: w for (lt, w) in dict_losses.items()}

    def forward(self, pred, gt, valid_mask=None):
        loss = 0
        for name in self.dict_losses:
            loss += self.dict_weights[name] * self.dict_losses[name](pred, gt, valid_mask)
        return loss


def build_loss(losses: dict, loss_kwargs: dict | None = None) -> MultiLoss:
    """Build a MultiLoss from a ``{"SIGLOSS": 1.0}`` / ``{"CE": 1.0}``-style config dict."""
    return MultiLoss({LossType[name]: weight for name, weight in losses.items()}, loss_kwargs)
