"""
The depth probe: a dense linear projection on top of frozen V-JEPA 2.1 final-layer
features, in the spirit of the DINOv3 / V-JEPA 2.1 evaluation protocol.

Pipeline:  features (B, D, gh, gw)
             -> LinearDepthHead  (1x1 conv -> n_bins logits)
             -> FeaturesToDepth  (AdaBins soft-argmax over depth bins -> metric depth)
             -> depth (B, 1, gh, gw)

The head is fully-convolutional, so it is agnostic to the patch-grid size: train on
KITTI's 24x78 grid, run on any other grid (CARLA) with no code change. Upsampling the
prediction to the ground-truth resolution happens in the training/eval loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearDepthHead(nn.Module):
    """Optional BatchNorm + 1x1 conv mapping patch features to per-bin logits.

    Mirrors the DINOv3 ``LinearHead.conv_depth`` (single final layer, no CLS token).
    """

    def __init__(self, in_channels: int, n_bins: int, use_batchnorm: bool = True):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(in_channels) if use_batchnorm else nn.Identity()
        self.conv_depth = nn.Conv2d(in_channels, n_bins, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.conv_depth.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.conv_depth.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, gh, gw) -> logits: (B, n_bins, gh, gw)
        return self.conv_depth(self.batchnorm(x))


class FeaturesToDepth(nn.Module):
    """Convert per-bin logits to metric depth via AdaBins soft-argmax.

    Copied (log/linear branch only) from probes/depth/dinov3_depth/models/__init__.py.
    If ``n_bins == 1`` it falls back to plain regression (relu(x) + min_depth).
    """

    def __init__(
        self,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        bins_strategy: str = "log",
        norm_strategy: str = "linear",
    ):
        super().__init__()
        assert bins_strategy in ("linear", "log"), "bins_strategy must be 'linear' or 'log'"
        assert norm_strategy in ("linear", "softmax", "sigmoid"), (
            "norm_strategy must be 'linear', 'softmax' or 'sigmoid'"
        )
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_bins = x.shape[1]
        if n_bins == 1:
            return torch.relu(x) + self.min_depth

        if self.bins_strategy == "linear":
            bins = torch.linspace(self.min_depth, self.max_depth, n_bins, device=x.device)
        else:  # "log"
            bins = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(self.min_depth)),
                    torch.log(torch.tensor(self.max_depth)),
                    n_bins,
                )
            ).to(x.device)

        if self.norm_strategy == "linear":
            logit = torch.relu(x) + 0.1
            logit = logit / logit.sum(dim=1, keepdim=True)
        elif self.norm_strategy == "softmax":
            logit = torch.softmax(x, dim=1)
        else:  # "sigmoid"
            logit = torch.sigmoid(x)
            logit = logit / logit.sum(dim=1, keepdim=True)

        depth = torch.einsum("ikmn,k->imn", logit, bins).unsqueeze(1)  # (B, 1, gh, gw)
        return depth


class DepthProbe(nn.Module):
    """Linear head + AdaBins depth conversion. Encoder lives offline (cached features)."""

    def __init__(
        self,
        embed_dim: int = 1024,
        n_bins: int = 256,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        bins_strategy: str = "log",
        norm_strategy: str = "linear",
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.head = LinearDepthHead(embed_dim, n_bins, use_batchnorm=use_batchnorm)
        self.features_to_depth = FeaturesToDepth(
            min_depth=min_depth,
            max_depth=max_depth,
            bins_strategy=bins_strategy,
            norm_strategy=norm_strategy,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, D, gh, gw) -> depth: (B, 1, gh, gw)
        return self.features_to_depth(self.head(features))

    @classmethod
    def from_config(cls, cfg) -> "DepthProbe":
        return cls(
            embed_dim=cfg.embed_dim,
            n_bins=cfg.n_bins,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            bins_strategy=cfg.bins_strategy,
            norm_strategy=cfg.norm_strategy,
            use_batchnorm=cfg.use_batchnorm,
        )
