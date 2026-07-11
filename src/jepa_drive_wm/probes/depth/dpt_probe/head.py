"""4-layer DPT head over stacked hierarchical V-JEPA 2.1 features.

Input is the 4-layer tap ``(B, L=4, D, gh, gw)`` -- one patch grid per tapped layer
([2,5,8,11] for ViT-B). The DPT head (``dpt_head.DPTHead``, the DINOv3 protocol)
reassembles each layer to a different scale, fuses them coarse-to-fine, and emits
``n_bins`` per-pixel logits at full resolution (24x78 grid -> 384x1248), which the
depth-bin conversion turns into metric depth.

``DptDepthProbe`` wraps the head + the depth-bin conversion (``_core.binning``).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .._core.binning import FeaturesToDepth
from .dpt_head import DPTHead


class DPTDepthHead(nn.Module):
    """4-layer DPT head (DINOv3 protocol) over stacked hierarchical features.

    V-JEPA has no CLS token, so ``readout_type="ignore"``: the reassemble blocks never
    touch the (here zero) readout token. ``use_batchnorm=False`` keeps the head off
    ``SyncBatchNorm`` (which needs a process group; would crash a single-GPU run).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_bins: int = 256,
        n_layers: int = 4,
        channels: int = 256,
        post_process_channels: tuple[int, ...] = (128, 256, 512, 1024),
        readout_type: str = "ignore",
        use_batchnorm: bool = False,
    ):
        super().__init__()
        if n_layers != 4:
            raise ValueError(f"DPTHead's reassemble stage is fixed to 4 layers; got n_layers={n_layers}.")
        self.n_layers = n_layers
        self.dpt = DPTHead(
            in_channels=tuple(embed_dim for _ in range(n_layers)),
            channels=channels,
            post_process_channels=list(post_process_channels),
            readout_type=readout_type,
            n_output_channels=n_bins,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, L, D, gh, gw) -> logits: (B, n_bins, H, W)
        if features.ndim != 5:
            raise ValueError(
                f"DPTDepthHead expects (B, L, D, gh, gw) hierarchical features; got "
                f"shape {tuple(features.shape)}."
            )
        if features.shape[1] != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} layers, got {features.shape[1]}.")
        b, _, d = features.shape[0], features.shape[1], features.shape[2]
        # DPTHead wants a list of [patch_map, readout_token] per layer; readout is ignored
        # (readout_type='ignore'), so pass a zero placeholder.
        zero_readout = features.new_zeros((b, d))
        inputs = [[features[:, i], zero_readout] for i in range(self.n_layers)]
        return self.dpt(inputs)


class DptDepthProbe(nn.Module):
    """4-layer DPT head + fixed-bin depth conversion over frozen hierarchical features.

    The head emits ``n_bins`` per-pixel logits; ``FeaturesToDepth`` soft-argmaxes them over
    fixed metric log-bins ``[bin_min_depth, bin_max_depth]`` -> metric depth.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_bins: int = 256,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        bins_strategy: str = "log",
        norm_strategy: str = "linear",
        bin_min_depth: float | None = None,
        bin_max_depth: float | None = None,
        n_layers: int = 4,
        dpt_channels: int = 256,
        dpt_post_process_channels: tuple[int, ...] = (128, 256, 512, 1024),
        dpt_readout: str = "ignore",
        dpt_use_batchnorm: bool = False,
    ):
        super().__init__()
        self.head = DPTDepthHead(
            embed_dim=embed_dim, n_bins=n_bins, n_layers=n_layers, channels=dpt_channels,
            post_process_channels=dpt_post_process_channels, readout_type=dpt_readout,
            use_batchnorm=dpt_use_batchnorm,
        )
        self.features_to_depth = FeaturesToDepth(
            min_depth=min_depth, max_depth=max_depth, bins_strategy=bins_strategy,
            norm_strategy=norm_strategy, bin_min_depth=bin_min_depth, bin_max_depth=bin_max_depth,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, L, D, gh, gw) -> depth (B, 1, H, W) at full resolution
        logits = self.head(features)
        return self.features_to_depth(logits)

    @classmethod
    def from_config(cls, cfg) -> "DptDepthProbe":
        return cls(
            embed_dim=cfg.head_embed_dim,   # 384 in layer_mode="pred", else 768
            n_bins=cfg.n_bins,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            bins_strategy=cfg.bins_strategy,
            norm_strategy=cfg.norm_strategy,
            bin_min_depth=cfg.bin_min_depth,
            bin_max_depth=cfg.bin_max_depth,
            n_layers=cfg.n_layers,
            dpt_channels=cfg.dpt_channels,
            dpt_post_process_channels=cfg.dpt_post_process_channels,
            dpt_readout=cfg.dpt_readout,
            dpt_use_batchnorm=cfg.dpt_use_batchnorm,
        )


class LinearDepthProbe(nn.Module):
    """DINOv3-style dense linear depth probe over the frozen final-layer patch grid.

    ``BatchNorm2d -> 1x1 Conv -> fixed-bin soft-argmax`` on ``(B, D, gh, gw)`` final tokens —
    the canonical V-JEPA 2.1 / DINOv3 dense-eval head (no feature pyramid, no multi-layer
    fusion). The only trainable parameters are the BatchNorm + the 1x1 conv (~198k for
    D=768, n_bins=256); depth comes out at the patch grid and is upsampled by the caller.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_bins: int = 256,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        bins_strategy: str = "log",
        norm_strategy: str = "linear",
        bin_min_depth: float | None = None,
        bin_max_depth: float | None = None,
    ):
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dim)
        self.proj = nn.Conv2d(embed_dim, n_bins, kernel_size=1)
        self.features_to_depth = FeaturesToDepth(
            min_depth=min_depth, max_depth=max_depth, bins_strategy=bins_strategy,
            norm_strategy=norm_strategy, bin_min_depth=bin_min_depth, bin_max_depth=bin_max_depth,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, D, gh, gw) final-layer grid -> depth (B, 1, gh, gw)
        if features.ndim != 4:
            raise ValueError(
                f"LinearDepthProbe expects a single (B, D, gh, gw) final grid; got "
                f"shape {tuple(features.shape)}. (layer_mode='quad' is DPT-only.)"
            )
        logits = self.proj(self.norm(features))
        return self.features_to_depth(logits)

    @classmethod
    def from_config(cls, cfg) -> "LinearDepthProbe":
        return cls(
            embed_dim=cfg.head_embed_dim,   # 384 in layer_mode="pred", else 768
            n_bins=cfg.n_bins,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            bins_strategy=cfg.bins_strategy,
            norm_strategy=cfg.norm_strategy,
            bin_min_depth=cfg.bin_min_depth,
            bin_max_depth=cfg.bin_max_depth,
        )


def build_probe(cfg) -> nn.Module:
    """Depth probe selected by ``cfg.head_type`` ("dpt" or "linear").

    ``getattr`` default keeps pre-``head_type`` checkpoints (pickled without the field) loadable
    as DPT.
    """
    head_type = getattr(cfg, "head_type", "dpt")
    if head_type == "linear":
        return LinearDepthProbe.from_config(cfg)
    if head_type == "dpt":
        return DptDepthProbe.from_config(cfg)
    raise ValueError(f"head_type must be 'dpt' or 'linear', got {head_type!r}")
