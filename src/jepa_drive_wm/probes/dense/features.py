"""Frozen V-JEPA 2.1 final-grid feature extraction (shared by all dense tasks).

Both depth and semseg consume the *same* input: the post-final-norm final-layer patch grid
``(B, D, gh, gw)`` (``extract_hierarchical(x)[:, -1]`` -- the wrapper guarantees the last tap
equals the standard final-normed output). ``feat_mode="pred"`` applies the frozen
``predictor_embed`` (768 -> 384) instead.
"""
from __future__ import annotations

import torch


def build_wrapper(cfg, image_hw):
    from jepa_drive_wm.utils.vjepa_wrapper import VJEPA21Size, VJEPA21Wrapper
    h, w = image_hw
    return VJEPA21Wrapper(size=VJEPA21Size[cfg.vjepa_size], image_height=h, image_width=w, verbose=True)


def make_featurizer(wrapper, cfg):
    """Return ``featurize(x_bchw) -> (B, D, gh, gw)`` final grid for the head."""
    def featurize(x: torch.Tensor) -> torch.Tensor:
        final = wrapper.extract_hierarchical(x)[:, -1]       # (B, 768, gh, gw), final-normed
        if cfg.feat_mode == "pred":
            final = wrapper.compress_final(final)            # (B, 384, gh, gw)
        return final
    return featurize
