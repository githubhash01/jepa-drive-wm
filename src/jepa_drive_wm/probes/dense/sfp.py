"""SimpleFeaturePyramid dense-prediction head over a single frozen final-layer grid.

The one architecture shared by depth and semseg. It takes the single final grid
``(B, D, gh, gw)`` and builds a 4-scale feature pyramid *from that one map* (ViTDet
SimpleFeaturePyramid / DPT-single-layer): parallel 1x1 projections + resampling to
{4x, 2x, 1x, 0.5x}, then coarse-to-fine fusion (``FeatureFusionBlock``), then an
``UpConvHead`` emitting ``n_out`` channels. ``n_out = n_bins`` (depth) or ``num_classes``
(semseg). Feeding one map through this pyramid is the same computation as the previous DPT
head fed 4 copies of the final layer -- it reproduces that head's result, honestly expressed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvModule, FeatureFusionBlock, UpConvHead


class SimpleFeaturePyramid(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        n_out: int = 256,
        channels: int = 256,
        pyramid_channels: tuple[int, ...] = (128, 256, 512, 1024),
        n_hidden_channels: int = 32,
        input_norm: bool = True,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(embed_dim) if input_norm else nn.Identity()
        # One map -> 4 pyramid levels: project to each width, then resample to a distinct scale.
        self.projects = nn.ModuleList(
            [ConvModule(embed_dim, c, kernel_size=1, act_cfg=None) for c in pyramid_channels]
        )
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(pyramid_channels[0], pyramid_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(pyramid_channels[1], pyramid_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(pyramid_channels[3], pyramid_channels[3], kernel_size=3, stride=2, padding=1),
        ])
        self.convs = nn.ModuleList(
            [ConvModule(c, channels, kernel_size=3, padding=1, act_cfg=None, bias=False) for c in pyramid_channels]
        )
        act_cfg = {"type": "ReLU"}
        self.fusion_blocks = nn.ModuleList(
            [FeatureFusionBlock(channels, act_cfg, None, bias=False) for _ in pyramid_channels]
        )
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = ConvModule(channels, channels, kernel_size=3, padding=1)
        self.head = UpConvHead(channels, n_out, n_hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, gh, gw) -> (B, n_out, H', W')
        if x.ndim != 4:
            raise ValueError(f"SimpleFeaturePyramid expects (B, D, gh, gw); got {tuple(x.shape)}.")
        x = self.input_norm(x)
        feats = [self.resize_layers[i](self.projects[i](x)) for i in range(len(self.projects))]
        feats = [self.convs[i](f) for i, f in enumerate(feats)]
        out = self.fusion_blocks[0](feats[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, feats[-(i + 1)])
        out = self.project(out)
        return self.head(out)

    @classmethod
    def from_config(cls, cfg) -> "SimpleFeaturePyramid":
        return cls(
            embed_dim=cfg.embed_dim_for_head,
            n_out=cfg.n_out,
            channels=cfg.sfp_channels,
            pyramid_channels=tuple(cfg.sfp_pyramid_channels),
        )
