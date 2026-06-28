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


class _UpBlock(nn.Module):
    """2x bilinear upsample followed by two 3x3 conv-norm-GELU layers."""

    def __init__(self, in_ch: int, out_ch: int, use_batchnorm: bool = True):
        super().__init__()
        norm = (lambda c: nn.BatchNorm2d(c)) if use_batchnorm else (lambda c: nn.Identity())
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = norm(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class ConvDepthDecoder(nn.Module):
    """Convolutional decoder with learned progressive upsampling (DPT-style, single layer).

    Unlike the 1x1 LinearDepthHead, this learns spatial structure: it projects the patch
    features, then upsamples ``n_upsample`` times (each 2x, with 3x3 convs) so the head can
    sharpen edges instead of emitting flat per-patch blocks. Operates purely on the final
    encoder feature map (B, D, gh, gw); no intermediate-layer access.
    """

    def __init__(
        self,
        in_channels: int,
        n_bins: int,
        channels: int = 256,
        n_upsample: int = 3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        norm = (lambda c: nn.BatchNorm2d(c)) if use_batchnorm else (lambda c: nn.Identity())
        self.proj = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.proj_norm = norm(channels)
        self.act = nn.GELU()

        blocks = []
        c = channels
        for _ in range(n_upsample):
            out_c = max(c // 2, 64)
            blocks.append(_UpBlock(c, out_c, use_batchnorm=use_batchnorm))
            c = out_c
        self.blocks = nn.ModuleList(blocks)

        # Final per-pixel bin logits. Small init keeps early depths near the bin mean.
        self.conv_depth = nn.Conv2d(c, n_bins, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_depth.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.conv_depth.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, gh, gw) -> logits: (B, n_bins, gh*2^n, gw*2^n)
        x = self.act(self.proj_norm(self.proj(x)))
        for block in self.blocks:
            x = block(x)
        return self.conv_depth(x)


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
    """Decoder head + AdaBins depth conversion. Encoder lives offline (cached features).

    ``head_type``:
      * "linear" — 1x1 conv on the patch grid (fast baseline, no spatial refinement)
      * "conv"   — convolutional decoder with learned upsampling (sharper, stronger)
    Both read only the final encoder feature map (B, D, gh, gw).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_bins: int = 256,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        bins_strategy: str = "log",
        norm_strategy: str = "linear",
        use_batchnorm: bool = True,
        head_type: str = "conv",
        decoder_channels: int = 256,
        decoder_upsample: int = 3,
    ):
        super().__init__()
        if head_type == "linear":
            self.head: nn.Module = LinearDepthHead(embed_dim, n_bins, use_batchnorm=use_batchnorm)
        elif head_type == "conv":
            self.head = ConvDepthDecoder(
                embed_dim, n_bins,
                channels=decoder_channels,
                n_upsample=decoder_upsample,
                use_batchnorm=use_batchnorm,
            )
        else:
            raise ValueError(f"head_type must be 'linear' or 'conv', got {head_type!r}")

        self.features_to_depth = FeaturesToDepth(
            min_depth=min_depth,
            max_depth=max_depth,
            bins_strategy=bins_strategy,
            norm_strategy=norm_strategy,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, D, gh, gw) -> depth: (B, 1, h, w) at the decoder's output resolution
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
            head_type=cfg.head_type,
            decoder_channels=cfg.decoder_channels,
            decoder_upsample=cfg.decoder_upsample,
        )
