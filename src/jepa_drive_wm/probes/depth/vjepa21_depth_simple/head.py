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

from .dpt_head import DPTHead


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

    Bin distribution follows probes/depth/dinov3_depth: "log" (scale-invariant / SID),
    "linear" (uniform), or "mixlog" (DINOv3's log->linear blend). The bin *range*
    (``bin_min_depth``, ``bin_max_depth``) is separate from the valid-mask floor so it can
    be set to the dataset's actual depth span (KITTI ~2.2-80m) instead of wasting bins
    below the nearest real pixel. If ``n_bins == 1`` it falls back to plain regression.
    """

    def __init__(
        self,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        bins_strategy: str = "log",
        norm_strategy: str = "linear",
        bin_min_depth: float | None = None,
        bin_max_depth: float | None = None,
    ):
        super().__init__()
        assert bins_strategy in ("linear", "log", "mixlog"), "bins_strategy must be 'linear', 'log' or 'mixlog'"
        assert norm_strategy in ("linear", "softmax", "sigmoid"), (
            "norm_strategy must be 'linear', 'softmax' or 'sigmoid'"
        )
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy
        # Bin range defaults to the valid-depth range if not given (backward compatible).
        self.bin_min_depth = float(bin_min_depth) if bin_min_depth is not None else min_depth
        self.bin_max_depth = float(bin_max_depth) if bin_max_depth is not None else max_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_bins = x.shape[1]
        if n_bins == 1:
            return torch.relu(x) + self.min_depth

        lo, hi = self.bin_min_depth, self.bin_max_depth
        if self.bins_strategy == "linear":
            bins = torch.linspace(lo, hi, n_bins, device=x.device)
        elif self.bins_strategy == "log":
            bins = torch.exp(
                torch.linspace(torch.log(torch.tensor(lo)), torch.log(torch.tensor(hi)), n_bins)
            ).to(x.device)
        else:  # "mixlog": DINOv3's per-index blend of log (near) -> linear (far)
            lin = torch.linspace(lo, hi, n_bins, device=x.device)
            log = torch.exp(
                torch.linspace(torch.log(torch.tensor(lo)), torch.log(torch.tensor(hi)), n_bins)
            ).to(x.device)
            t = torch.linspace(1.0, 0.0, n_bins, device=x.device)
            bins = t * log + (1.0 - t) * lin

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


class DPTDepthHead(nn.Module):
    """4-layer DPT head (DINOv3 protocol) over stacked hierarchical V-JEPA features.

    Input is the hierarchical cache ``(B, L, D, gh, gw)`` — one patch grid per tapped
    layer ([2,5,8,11] for ViT-B). The DPT head reassembles each layer to a different
    scale, fuses them coarse-to-fine, and emits ``n_bins`` per-pixel logits at full
    resolution (24x78 grid -> 384x1248), which ``FeaturesToDepth`` turns into depth.

    V-JEPA has no CLS token, so ``readout_type="ignore"``: the DPT head's reassemble
    blocks never touch the (here zero) readout token. ``use_batchnorm=False`` keeps the
    head off ``torch.nn.SyncBatchNorm`` (which needs a process group; would crash a
    single-GPU run).
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
            raise ValueError(
                f"DPTHead's reassemble stage is fixed to 4 layers; got n_layers={n_layers}."
            )
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
                f"shape {tuple(features.shape)}. Use the 'vjepa_vitb_hier' cache."
            )
        if features.shape[1] != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} layers, got {features.shape[1]}."
            )
        b, _, d = features.shape[0], features.shape[1], features.shape[2]
        # DPTHead wants a list of [patch_map, readout_token] per layer; readout is
        # ignored (readout_type='ignore'), so pass a zero placeholder.
        zero_readout = features.new_zeros((b, d))
        inputs = [[features[:, i], zero_readout] for i in range(self.n_layers)]
        return self.dpt(inputs)


class AdaptiveBins(nn.Module):
    """AdaBins-style per-image adaptive bin centers (Bhat et al., 2021).

    Instead of fixed bin centers, a small MLP reads a global image descriptor (mean-pooled
    frozen features) and predicts ``n_bins`` positive widths that partition [min, max]; the
    bin centers become per-image. Combined with the head's per-pixel logits in a soft-argmax,
    the bins concentrate on each image's actual depth range. (This is the MLP-regressor
    variant; the original also offers a mini-ViT — mean-pool + MLP is the lighter form.)
    """

    def __init__(self, in_dim: int, n_bins: int, min_depth: float, max_depth: float, hidden: int = 256):
        super().__init__()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.n_bins = n_bins
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        """global_feat: (B, in_dim) -> bin centers (B, n_bins), ascending in [min, max]."""
        w = self.mlp(global_feat.float())            # (B, n_bins) raw width logits
        w = torch.softmax(w, dim=1) + 1e-3           # positive normalised widths
        w = w / w.sum(dim=1, keepdim=True)
        w = w * (self.max_depth - self.min_depth)    # scale to the depth span
        # centers = left edge + half width; left edges are the cumulative sum of prior widths.
        left = self.min_depth + torch.cat(
            [torch.zeros_like(w[:, :1]), torch.cumsum(w[:, :-1], dim=1)], dim=1
        )
        centers = left + w / 2.0                     # (B, n_bins)
        return centers


def _softargmax_with_centers(logits: torch.Tensor, centers: torch.Tensor, norm_strategy: str) -> torch.Tensor:
    """Per-image soft-argmax: (B, n_bins, H, W) logits + (B, n_bins) centers -> (B, 1, H, W)."""
    if norm_strategy == "softmax":
        p = torch.softmax(logits, dim=1)
    elif norm_strategy == "sigmoid":
        p = torch.sigmoid(logits)
        p = p / p.sum(dim=1, keepdim=True)
    else:  # "linear"
        p = torch.relu(logits) + 0.1
        p = p / p.sum(dim=1, keepdim=True)
    depth = torch.einsum("bkhw,bk->bhw", p.float(), centers.float()).unsqueeze(1)
    return depth


class DepthProbe(nn.Module):
    """Decoder head + AdaBins depth conversion. Encoder lives offline (cached features).

    ``head_type``:
      * "linear" — 1x1 conv on the patch grid (fast baseline, no spatial refinement)
      * "conv"   — convolutional decoder with learned upsampling (sharper, stronger)
      * "dpt"    — 4-layer DPT head fusing intermediate layers (needs hierarchical cache)
    "linear"/"conv" read the final feature map (B, D, gh, gw); "dpt" reads (B, L, D, gh, gw).

    ``adaptive_bins=True`` predicts per-image bin centers (AdaBins) instead of fixed bins;
    the last predicted centers are stored on ``self.bin_centers`` for the Chamfer bin loss.
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
        use_batchnorm: bool = True,
        head_type: str = "conv",
        decoder_channels: int = 256,
        decoder_upsample: int = 3,
        n_layers: int = 4,
        dpt_channels: int = 256,
        dpt_post_process_channels: tuple[int, ...] = (128, 256, 512, 1024),
        dpt_readout: str = "ignore",
        dpt_use_batchnorm: bool = False,
        adaptive_bins: bool = False,
    ):
        super().__init__()
        self.head_type = head_type
        self.norm_strategy = norm_strategy
        self.adaptive_bins = adaptive_bins
        self.bin_centers: torch.Tensor | None = None  # last predicted centers (for chamfer loss)
        if head_type == "linear":
            self.head: nn.Module = LinearDepthHead(embed_dim, n_bins, use_batchnorm=use_batchnorm)
        elif head_type == "conv":
            self.head = ConvDepthDecoder(
                embed_dim, n_bins,
                channels=decoder_channels,
                n_upsample=decoder_upsample,
                use_batchnorm=use_batchnorm,
            )
        elif head_type == "dpt":
            self.head = DPTDepthHead(
                embed_dim=embed_dim,
                n_bins=n_bins,
                n_layers=n_layers,
                channels=dpt_channels,
                post_process_channels=dpt_post_process_channels,
                readout_type=dpt_readout,
                use_batchnorm=dpt_use_batchnorm,
            )
        else:
            raise ValueError(f"head_type must be 'linear', 'conv', or 'dpt', got {head_type!r}")

        self.features_to_depth = FeaturesToDepth(
            min_depth=min_depth,
            max_depth=max_depth,
            bins_strategy=bins_strategy,
            norm_strategy=norm_strategy,
            bin_min_depth=bin_min_depth,
            bin_max_depth=bin_max_depth,
        )

        if adaptive_bins:
            # Global descriptor = mean-pooled features across the patch grid. For the DPT
            # head that's all L layers concatenated (L*D); otherwise the single map (D).
            in_dim = (n_layers * embed_dim) if head_type == "dpt" else embed_dim
            self.adaptive = AdaptiveBins(
                in_dim=in_dim, n_bins=n_bins,
                min_depth=self.features_to_depth.bin_min_depth,
                max_depth=self.features_to_depth.bin_max_depth,
            )

    def _global_descriptor(self, features: torch.Tensor) -> torch.Tensor:
        # (B, L, D, gh, gw) -> (B, L*D) ; or (B, D, gh, gw) -> (B, D)
        if features.ndim == 5:
            return features.mean(dim=(3, 4)).flatten(1)
        return features.mean(dim=(2, 3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features -> depth (B, 1, h, w) at the decoder's output resolution
        logits = self.head(features)
        if self.adaptive_bins:
            centers = self.adaptive(self._global_descriptor(features))  # (B, n_bins)
            self.bin_centers = centers
            return _softargmax_with_centers(logits, centers, self.norm_strategy)
        return self.features_to_depth(logits)

    @classmethod
    def from_config(cls, cfg) -> "DepthProbe":
        return cls(
            embed_dim=cfg.embed_dim,
            n_bins=cfg.n_bins,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            bins_strategy=cfg.bins_strategy,
            norm_strategy=cfg.norm_strategy,
            bin_min_depth=getattr(cfg, "bin_min_depth", None),
            bin_max_depth=getattr(cfg, "bin_max_depth", None),
            use_batchnorm=cfg.use_batchnorm,
            head_type=cfg.head_type,
            decoder_channels=cfg.decoder_channels,
            decoder_upsample=cfg.decoder_upsample,
            n_layers=cfg.n_layers,
            dpt_channels=cfg.dpt_channels,
            dpt_post_process_channels=cfg.dpt_post_process_channels,
            dpt_readout=cfg.dpt_readout,
            dpt_use_batchnorm=cfg.dpt_use_batchnorm,
            adaptive_bins=getattr(cfg, "adaptive_bins", False),
        )
