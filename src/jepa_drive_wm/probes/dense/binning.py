"""Depth-bin conversion (depth task only).

A depth head emits ``n_bins`` per-pixel logits; ``FeaturesToDepth`` turns them into metric
depth by a soft-argmax over fixed depth bins spanning ``[bin_min_depth, bin_max_depth]`` under a
"log"/"linear"/"mixlog" law.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeaturesToDepth(nn.Module):
    """Convert per-bin logits to metric depth via a fixed-bin soft-argmax.

    Bin distribution: "log" (SID), "linear" (uniform), or "mixlog" (log->linear blend). The bin
    range (``bin_min_depth``, ``bin_max_depth``) is separate from the valid-mask floor so it can
    target the dataset's actual depth span. ``n_bins == 1`` falls back to plain regression.
    """

    def __init__(self, min_depth=0.001, max_depth=80.0, bins_strategy="log",
                 norm_strategy="linear", bin_min_depth=None, bin_max_depth=None):
        super().__init__()
        assert bins_strategy in ("linear", "log", "mixlog")
        assert norm_strategy in ("linear", "softmax", "sigmoid")
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy
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
        else:  # "mixlog"
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
