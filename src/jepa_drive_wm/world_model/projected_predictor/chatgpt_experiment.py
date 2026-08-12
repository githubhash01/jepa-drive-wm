"""Frame-wise V-JEPA 2.1 ViT-B predictor experiment for non-square warped images.

This module is deliberately dataset-agnostic. It consumes:

    warped_rgb:  [H, W, 3] in [0, 1]
    patch_valid: [H/16, W/16] bool

and runs the released V-JEPA 2.1 ViT-B encoder/predictor in *image mode*.
For a 384x1248 frame, the token grid is 1x24x78.

Important checkpoint fact
-------------------------
The released distilled ViT-B predictor accepts 768-D ViT-B context tokens but
projects its outputs to the 1664-D ViT-G teacher space. Consequently this file
provides a faithful zero-shot diagnostic, not a direct 768-D completion head.
For post-training, rebuild the two output heads at 768 dimensions and load the
remaining predictor weights from the checkpoint.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_drive_wm.paths import VJEPA_REPO

if str(VJEPA_REPO) not in sys.path:
    sys.path.insert(0, str(VJEPA_REPO))

from app.vjepa_2_1.models.predictor import vit_predictor
from app.vjepa_2_1.models.vision_transformer import vit_base


VJEPA_MEAN = (0.485, 0.456, 0.406)
VJEPA_STD = (0.229, 0.224, 0.225)


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


def inspect_vitb_checkpoint(checkpoint_path: str | Path) -> None:
    """Print the predictor interface encoded by a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pred = _clean_state_dict(checkpoint["predictor"])

    print("checkpoint keys:", sorted(checkpoint.keys()))
    print("predictor tensors relevant to the interface:")
    for key in sorted(pred):
        if (
            "predictor_embed" in key
            or "predictor_proj" in key
            or "mask_tokens" in key
        ):
            print(f"  {key:55s} {tuple(pred[key].shape)}")


def build_released_vitb_384(
    checkpoint_path: str | Path,
    device: torch.device | str = "cuda",
) -> tuple[nn.Module, nn.Module]:
    """Build and strictly load the released distilled ViT-B/16-384 pair.

    The model is constructed at its original square pretraining resolution and
    temporal capacity. The encoder can nevertheless receive a non-square image,
    and ``StaticNonSquarePredictor`` supplies the predictor with the runtime
    24x78 grid explicitly.
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
        interpolate_rope=True,
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


def rgb_to_vjepa_image(rgb: torch.Tensor) -> torch.Tensor:
    """Convert [H,W,3] or [B,H,W,3] RGB in [0,1] to [B,3,1,H,W]."""
    rgb = torch.as_tensor(rgb)
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected [H,W,3] or [B,H,W,3], got {tuple(rgb.shape)}")

    x = rgb.float().permute(0, 3, 1, 2).unsqueeze(2)
    mean = x.new_tensor(VJEPA_MEAN).view(1, 3, 1, 1, 1)
    std = x.new_tensor(VJEPA_STD).view(1, 3, 1, 1, 1)
    return (x - mean) / std


def complementary_patch_masks(
    patch_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make sorted row-major context and target index tensors for one image."""
    patch_valid = torch.as_tensor(patch_valid, dtype=torch.bool)
    if patch_valid.ndim != 2:
        raise ValueError(
            "The initial experiment intentionally handles one future image at a "
            f"time; expected [Gh,Gw], got {tuple(patch_valid.shape)}"
        )

    flat = patch_valid.reshape(-1)
    context = torch.nonzero(flat, as_tuple=False).flatten().unsqueeze(0)
    target = torch.nonzero(~flat, as_tuple=False).flatten().unsqueeze(0)
    if context.numel() == 0:
        raise ValueError("No valid context patches.")
    if target.numel() == 0:
        raise ValueError("No missing target patches; the predictor has nothing to fill.")
    return context.long(), target.long()


class StaticNonSquarePredictor(nn.Module):
    """Run an unmodified predictor's weights with the correct runtime grid.

    The upstream predictor forwards only token indices to each RoPE block. For a
    non-square 24x78 grid, it must also pass H_patches and W_patches; otherwise
    the indices are decoded using the square pretraining grid. This adapter keeps
    all learned modules untouched and reproduces the upstream forward pass while
    supplying the missing geometry.
    """

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        grid_hw: tuple[int, int],
        mask_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        p = self.predictor
        if not p.use_rope:
            raise NotImplementedError("This adapter is intentionally RoPE-only.")
        if p.mask_tokens is None or p.num_mask_tokens == 0:
            raise ValueError("The predictor was not constructed with mask tokens.")
        if p.chop_last_n_tokens != 0:
            raise NotImplementedError("chop_last_n_tokens is not supported here.")

        if context_tokens.ndim != 3:
            raise ValueError("context_tokens must be [B,N_context,D].")
        B = context_tokens.shape[0]
        if context_indices.shape[0] != B or target_indices.shape[0] != B:
            raise ValueError("Mask batch size must match context token batch size.")

        Gh, Gw = grid_hw
        total_patches = Gh * Gw
        all_indices = torch.cat([context_indices, target_indices], dim=1)
        if int(all_indices.min()) < 0 or int(all_indices.max()) >= total_patches:
            raise ValueError("Patch index falls outside the supplied token grid.")
        if all_indices.shape[1] != total_patches:
            raise ValueError("Context and target masks must partition the full grid.")
        if torch.unique(all_indices[0]).numel() != total_patches:
            raise ValueError("Context and target masks overlap or omit an index.")

        x = p.predictor_embed(context_tokens)
        n_context = x.shape[1]

        token = p.mask_tokens[mask_index % p.num_mask_tokens]
        target_tokens = token.expand(B, total_patches, -1)
        target_tokens = torch.gather(
            target_tokens,
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, target_tokens.shape[-1]),
        )

        x = torch.cat([x, target_tokens], dim=1)
        masks = torch.cat([context_indices, target_indices], dim=1)

        # Put all tokens into spatial row-major order for attention.
        order = torch.argsort(masks, dim=1)
        masks_sorted = torch.gather(masks, 1, order)
        x = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        if p.modality_embedding:
            x = x + p.img_mod_embed

        for block in p.predictor_blocks:
            x, _ = block(
                x,
                mask=masks_sorted,
                T=1,
                H_patches=Gh,
                W_patches=Gw,
                mode="image",
            )
        x = p.predictor_norm(x)

        # Restore [context tokens, target tokens] ordering used by the API.
        inverse_order = torch.argsort(order, dim=1)
        x = torch.gather(
            x, 1, inverse_order.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        context_hidden = x[:, :n_context]
        target_hidden = x[:, n_context:]

        target_output = p.predictor_proj(target_hidden)
        context_output = None
        if p.return_all_tokens:
            context_output = p.predictor_proj_context(context_hidden)
        return target_output, context_output


@dataclass
class StaticZeroShotOutput:
    context_indices: torch.Tensor       # [1,N_context]
    target_indices: torch.Tensor        # [1,N_target]
    context_student: torch.Tensor       # [1,N_context,768]
    target_teacher: torch.Tensor        # [1,N_target,1664]
    context_teacher: torch.Tensor       # [1,N_context,1664]
    dense_teacher: torch.Tensor         # [1,Gh*Gw,1664]


class StaticVJEPAZeroShot(nn.Module):
    """Faithful frame-wise zero-shot predictor experiment."""

    def __init__(self, encoder: nn.Module, predictor: nn.Module, patch_size: int = 16):
        super().__init__()
        self.encoder = encoder
        self.static_predictor = StaticNonSquarePredictor(predictor)
        self.patch_size = int(patch_size)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @torch.no_grad()
    def encode_full(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb_to_vjepa_image(rgb).to(self.device)
        return self.encoder(x, training=False)

    @torch.no_grad()
    def forward(
        self,
        warped_rgb: torch.Tensor,
        patch_valid: torch.Tensor,
        *,
        mask_index: int = 0,
    ) -> StaticZeroShotOutput:
        H, W = warped_rgb.shape[:2]
        if H % self.patch_size or W % self.patch_size:
            raise ValueError("Image dimensions must be divisible by patch_size.")
        grid_hw = (H // self.patch_size, W // self.patch_size)

        context_indices, target_indices = complementary_patch_masks(patch_valid)
        context_indices = context_indices.to(self.device)
        target_indices = target_indices.to(self.device)

        x = rgb_to_vjepa_image(warped_rgb).to(self.device)

        # Crucial: remove invalid patch tokens before encoder self-attention.
        context_student = self.encoder(
            x,
            masks=context_indices,
            training=False,
        )

        target_teacher, context_teacher = self.static_predictor(
            context_student,
            context_indices,
            target_indices,
            grid_hw=grid_hw,
            mask_index=mask_index,
        )
        if context_teacher is None:
            raise RuntimeError("Expected return_all_tokens=True in released ViT-B predictor.")

        dense_teacher = target_teacher.new_empty(
            (1, grid_hw[0] * grid_hw[1], target_teacher.shape[-1])
        )
        dense_teacher[:, context_indices[0]] = context_teacher
        dense_teacher[:, target_indices[0]] = target_teacher

        return StaticZeroShotOutput(
            context_indices=context_indices,
            target_indices=target_indices,
            context_student=context_student,
            target_teacher=target_teacher,
            context_teacher=context_teacher,
            dense_teacher=dense_teacher,
        )


@torch.no_grad()
def cross_affinity_correlation(
    output: StaticZeroShotOutput,
    gt_student_tokens: torch.Tensor,
) -> torch.Tensor:
    """Basis-independent diagnostic across the 1664-D/768-D space boundary.

    It compares missing-to-context cosine-similarity matrices rather than the
    feature vectors themselves. This is only a diagnostic; it is not a training
    loss and does not make the two representation spaces equivalent.
    """
    gt_context = gt_student_tokens[:, output.context_indices[0]]
    gt_target = gt_student_tokens[:, output.target_indices[0]]

    pred_similarity = F.normalize(output.target_teacher, dim=-1) @ F.normalize(
        output.context_teacher, dim=-1
    ).transpose(-1, -2)
    gt_similarity = F.normalize(gt_target, dim=-1) @ F.normalize(
        gt_context, dim=-1
    ).transpose(-1, -2)

    a = pred_similarity.flatten().float()
    b = gt_similarity.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b) / (a.norm() * b.norm()).clamp_min(1e-12)


if __name__ == "__main__":
    from jepa_drive_wm.data.kitti import KITTISequence
    from jepa_drive_wm.paths import VJEPA_CKPT_DIR
    from jepa_drive_wm.world_model.projected_predictor.warp_rgb import WarpModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = 384, 1248
    SEQ_NR, SOURCE_ID, FUTURE_ID = 9, 375, 380  # same clip as projected_predictor.py

    # Warp the source frame to the future view (same settings as projected_predictor.py).
    seq = KITTISequence(sequence_nr=SEQ_NR)
    rgb = torch.from_numpy(seq.get_image(SOURCE_ID)).float().to(device)
    depth = torch.from_numpy(seq.get_depth(SOURCE_ID)).float().to(device)
    orig_h, orig_w = rgb.shape[:2]
    rgb = F.interpolate(rgb.permute(2, 0, 1)[None], size=(H, W),
                        mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    depth = F.interpolate(depth[None, None], size=(H, W), mode="nearest")[0, 0]
    K = seq.calib.K2.clone()
    K[0] *= W / orig_w
    K[1] *= H / orig_h
    warper = WarpModule(intrinsics=K, image_size=(H, W), radius_px=1.0,
                        patch_size=16, patch_coverage_threshold=0.7).to(device)
    ego = torch.from_numpy(seq.get_ego_motion(SOURCE_ID, FUTURE_ID)).float().to(device)
    warp = warper(rgb=rgb, depth=depth, ego_motion=ego)

    encoder, predictor = build_released_vitb_384(
        VJEPA_CKPT_DIR / "vjepa2_1_vitb_dist_vitG_384.pt", device
    )
    zero_shot = StaticVJEPAZeroShot(encoder, predictor)

    output = zero_shot(warp.rgb_patch_masked.clamp(0, 1).cpu(), warp.patch_valid.cpu())
    print(f"context tokens: {output.context_indices.shape[1]}, "
          f"missing tokens: {output.target_indices.shape[1]}")

    # GT student tokens of the REAL future frame (image mode, full visibility).
    gt_rgb = torch.from_numpy(seq.get_image(FUTURE_ID)).float()
    gt_rgb = F.interpolate(gt_rgb.permute(2, 0, 1)[None], size=(H, W),
                           mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    gt_student = zero_shot.encode_full(gt_rgb)

    corr = cross_affinity_correlation(output, gt_student)
    print(f"cross-affinity correlation (pred vs GT): {corr.item():.4f}")

    # Floor: the same diagnostic with the SOURCE frame as "ground truth" tells us
    # how much of the correlation is explained by the scene simply not changing.
    src_student = zero_shot.encode_full(rgb.cpu())
    corr_src = cross_affinity_correlation(output, src_student)
    print(f"cross-affinity correlation (pred vs source frame): {corr_src.item():.4f}")
