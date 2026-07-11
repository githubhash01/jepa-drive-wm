"""Conv building blocks for the dense-prediction head.

Trimmed from the DINOv3 DPT head (previously ``dpt_probe/dpt_head.py``): the reusable pieces
that the ``SimpleFeaturePyramid`` fuses with — ``ConvModule``, ``PreActResidualConvUnit``,
``FeatureFusionBlock``, ``UpConvHead``, ``Interpolate``. ``SyncBatchNorm`` is replaced with
plain ``BatchNorm2d`` (single-GPU; SyncBN needs a process group). The DPT reassemble/DPTHead
are gone — the pyramid is built for a single input in ``sfp.py``.
"""
from __future__ import annotations

import torch
from torch import nn


def kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"):
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    """Conv (+ optional BatchNorm) (+ optional ReLU), with configurable op order.

    Matches the DPT head's ConvModule for the paths it uses (``norm_cfg=None`` -> no norm,
    ``act_cfg`` = ReLU or None). ``norm_cfg`` truthy inserts a ``BatchNorm2d`` (was SyncBN).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias="auto", norm_cfg=None, act_cfg=dict(type="ReLU"),
                 order=("conv", "norm", "act")):
        super().__init__()
        self.order = order
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.with_norm:
            norm_channels = out_channels if order.index("norm") > order.index("conv") else in_channels
            self.bn = nn.BatchNorm2d(norm_channels)
        if self.with_activation:
            self.activate = nn.ReLU()
        kaiming_init(self.conv)
        if self.with_norm:
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.bn(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class UpConvHead(nn.Module):
    """3-conv output head with one 2x upsample: features -> n_output_channels."""

    def __init__(self, features, n_output_channels, n_hidden_channels=32):
        super().__init__()
        self.n_output_channels = n_output_channels
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, n_hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_hidden_channels, n_output_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.head(x)


class PreActResidualConvUnit(nn.Module):
    """Pre-activation residual conv unit (two ``act->conv->norm`` blocks + skip)."""

    def __init__(self, in_channels, act_cfg, norm_cfg, stride=1, dilation=1, bias=False):
        super().__init__()
        self.conv1 = ConvModule(in_channels, in_channels, 3, stride=stride, padding=dilation,
                                dilation=dilation, norm_cfg=norm_cfg, act_cfg=act_cfg, bias=bias,
                                order=("act", "conv", "norm"))
        self.conv2 = ConvModule(in_channels, in_channels, 3, padding=1, norm_cfg=norm_cfg,
                                act_cfg=act_cfg, bias=bias, order=("act", "conv", "norm"))

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs


class FeatureFusionBlock(nn.Module):
    """Merge a coarse feature map with the next finer stage, then 2x upsample."""

    def __init__(self, in_channels, act_cfg, norm_cfg, align_corners=True, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.align_corners = align_corners
        self.out_channels = in_channels
        self.project = ConvModule(in_channels, self.out_channels, kernel_size=1, act_cfg=None, bias=True)
        self.res_conv_unit1 = PreActResidualConvUnit(in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg, bias=bias)
        self.res_conv_unit2 = PreActResidualConvUnit(in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg, bias=bias)

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = nn.functional.interpolate(inputs[1], size=(x.shape[2], x.shape[3]),
                                                mode="bilinear", align_corners=False)
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        return self.project(x)
