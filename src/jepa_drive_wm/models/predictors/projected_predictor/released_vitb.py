"""Loader for the released V-JEPA 2.1 distilled ViT-B/16-384 encoder+predictor.

RoPE position handling deliberately mirrors the released configuration in
``vjepa2/src/hub/backbones.py::_make_vjepa2_1_model``:

- encoder: ``interpolate_rope=True``.  The released encoder was distilled with
  normalized positions, so any H_patches x W_patches grid is rescaled onto the
  fixed pretraining span [0, 15] per spatial axis.
- predictor: ``interpolate_rope=False``.  The released 12-block predictor was
  trained on raw integer patch positions; enabling interpolation here would
  compress its learned local position metric (5x horizontally at 24x78).

On the non-square KITTI grid the raw-position predictor extrapolates width
offsets beyond its trained range; ``ProjectedPredictor`` fine-tunes the
predictor body, which adapts that far-field behaviour while keeping the local
geometry exactly as pretrained.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from jepa_drive_wm.paths import VJEPA_REPO

if str(VJEPA_REPO) not in sys.path:
    sys.path.insert(0, str(VJEPA_REPO))

from app.vjepa_2_1.models.predictor import vit_predictor
from app.vjepa_2_1.models.vision_transformer import vit_base


def _clean_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove DDP/wrapper prefixes used by the released training checkpoint."""
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[len("module.") :]
        while key.startswith("backbone."):
            key = key[len("backbone.") :]
        cleaned[key] = value
    return cleaned


def build_released_vitb_384(
    checkpoint_path: str | Path,
    device: torch.device | str = "cuda",
) -> tuple[nn.Module, nn.Module]:
    """Build and strictly load the released distilled ViT-B/16-384 pair.

    The model is constructed at its original square pretraining resolution and
    temporal capacity.  The encoder handles non-square inputs itself; the
    predictor relies on its caller (``ProjectedPredictor``) to pass the true
    ``T``/``H_patches``/``W_patches`` into every RoPE block.

    Both modules are returned frozen and in eval mode; the caller decides what
    to fine-tune.
    """
    device = torch.device(device)

    encoder = vit_base(
        img_size=(384, 384),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        handle_nonsquare_inputs=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
        # The released distilled ViT-B predictor consumes the final 768-D layer.
        n_output_distillation=1,
    )

    predictor = vit_predictor(
        img_size=(384, 384),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        num_heads=12,
        use_mask_tokens=True,
        num_mask_tokens=8,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_rope=True,
        interpolate_rope=False,
        img_temporal_dim_size=1,
        modality_embedding=True,
        n_output_distillation=1,
        teacher_embed_dim=1664,
        return_all_tokens=True,
        use_activation_checkpointing=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_key = "ema_encoder" if "ema_encoder" in checkpoint else "target_encoder"
    if encoder_key not in checkpoint:
        raise KeyError(
            "Checkpoint contains neither 'ema_encoder' nor 'target_encoder'. "
            f"Found: {sorted(checkpoint.keys())}"
        )

    encoder_msg = encoder.load_state_dict(
        _clean_state_dict(checkpoint[encoder_key]), strict=True
    )
    predictor_msg = predictor.load_state_dict(
        _clean_state_dict(checkpoint["predictor"]), strict=True
    )
    print(f"loaded encoder ({encoder_key}): {encoder_msg}")
    print(f"loaded predictor: {predictor_msg}")

    encoder.eval().to(device)
    predictor.eval().to(device)
    for module in (encoder, predictor):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    return encoder, predictor
