"""Geometry-guided residual V-JEPA 2.1 projected predictor.

The model deliberately keeps the learned part minimal:

    Z_hat_{i+1} = Z0_{i+1} + DeltaZ_{i+1}

where

    Z0_{i+1} = warped image-mode latent on valid geometric patches
               previous static latent on invalid geometric patches

and ``DeltaZ`` is produced directly from the normalized 384-D hidden state of
V-JEPA 2.1's pretrained ViT-B predictor through one new shared linear head:

    384 -> 768.

Five observed RGB frames are used to construct two staggered eight-frame video
streams.  With V-JEPA's tubelet_size=2, the staggered streams provide four
adjacent transition fields:

    Stream 2, tubelet 2 : t   -> t+1
    Stream 1, tubelet 2 : t+1 -> t+2
    Stream 2, tubelet 3 : t+2 -> t+3
    Stream 1, tubelet 3 : t+3 -> t+4

The full released predictor transformer is retained and post-trained; only its
original 384->1664 ViT-G output heads are bypassed.  The frozen ViT-B encoder is
used in both video mode (predictor context) and image mode (static latent space).

This module is intentionally per-sample for the first implementation because
geometry produces variable-length context/target masks.  A leading batch
axis of size one is nevertheless retained in neural-network outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_drive_wm.models.predictors.projected_predictor.stream_builder import StreamPair, VideoStream, VideoStreamBuilder
from jepa_drive_wm.models.predictors.projected_predictor.latent_proposal import LatentProposal


VJEPA_MEAN = (0.485, 0.456, 0.406)
VJEPA_STD = (0.229, 0.224, 0.225)


@dataclass
class ProjectedPredictorOutput:
    """Outputs required for training, evaluation, and visualisation.

    All latent tensors retain a leading batch dimension of one.

    Shapes
    ------
    predictions:
        ``(1, 4, N, 768)`` predicted static image-mode latents.
    proposals:
        ``(1, 4, N, 768)`` zero-learning warp+copy proposals used at each
        autoregressive step.
    corrections:
        ``(1, 4, N, 768)`` learned corrections added to the proposals.
    transport_hidden:
        ``(1, 4, N, 384)`` normalized pretrained-predictor hidden fields before
        the new correction projection.
    warped_latents:
        ``(1, 4, N, 768)`` image-mode encodings of the four masked RGB warps.
    patch_valid:
        ``(1, 4, N)`` geometric patch validity for each future frame.
    z_t:
        ``(1, N, 768)`` known static latent at the final observed frame.
    target_latents:
        Optional ``(1, 4, N, 768)`` ground-truth future image-mode latents,
        populated only when ``future_rgb`` is supplied.
    """

    predictions: torch.Tensor
    proposals: torch.Tensor
    corrections: torch.Tensor
    transport_hidden: torch.Tensor
    warped_latents: torch.Tensor
    patch_valid: torch.Tensor
    z_t: torch.Tensor
    target_latents: torch.Tensor | None
    streams: StreamPair


class ProjectedPredictor(nn.Module):
    """Projected predictor using pretrained V-JEPA temporal reasoning.

    Parameters
    ----------
    encoder:
        Released V-JEPA 2.1 ViT-B EMA encoder.  The same module is used in
        image mode (T=1) and video mode (T=8).  It is frozen by default.
    predictor:
        Released distilled ViT-B predictor.  Its pretrained input projection,
        mask tokens, 12 transformer blocks and final LayerNorm are reused.
        The original ViT-G output projection(s) are bypassed.
    warp_module:
        Deterministic RGB-D ``WarpModule`` used by ``VideoStreamBuilder``.
    patch_size:
        ViT patch size.  V-JEPA 2.1 ViT-B uses 16.
    image_dim:
        Static image-latent dimension.  ViT-B uses 768.
    predictor_dim:
        Predictor hidden width.  Released ViT-B uses 384.
    finetune_predictor:
        If True, post-train the pretrained predictor body.  The frozen image/
        video encoder remains fixed.  The obsolete 1664-D output heads remain
        frozen because they are not part of the forward graph.
    zero_init_correction:
        Zero-initialise the new 384->768 projection so that the model starts
        exactly at the deterministic warp+copy baseline.
    mask_index:
        Which pretrained learned mask-token parameter to use for missing
        tubelets.
    """

    NUM_FUTURE_STEPS = 4

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        warp_module: nn.Module,
        *,
        patch_size: int = 16,
        image_dim: int = 768,
        predictor_dim: int = 384,
        finetune_predictor: bool = True,
        zero_init_correction: bool = True,
        mask_index: int = 1,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.warp_module = warp_module
        self.stream_builder = VideoStreamBuilder(self.warp_module)
        self.latent_proposal = LatentProposal()

        self.patch_size = int(patch_size)
        self.image_dim = int(image_dim)
        self.predictor_dim = int(predictor_dim)
        self.finetune_predictor = bool(finetune_predictor)
        self.mask_index = int(mask_index)

        self._validate_predictor_interface()

        # The only newly introduced learned module.
        self.correction_head = nn.Linear(
            self.predictor_dim,
            self.image_dim,
            bias=True,
        )
        if zero_init_correction:
            nn.init.zeros_(self.correction_head.weight)
            nn.init.zeros_(self.correction_head.bias)

        # The frozen EMA encoder defines both the input feature basis and the
        # image-mode regression target used by existing dense decoders.
        self._set_requires_grad(self.encoder, False)
        self.encoder.eval()

        # Reuse/post-train the predictor body but never train the obsolete
        # ViT-G output heads, which are bypassed by this model.
        self._configure_predictor_gradients()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device of the frozen V-JEPA encoder."""
        try:
            return next(self.encoder.parameters()).device
        except StopIteration:
            return next(self.correction_head.parameters()).device

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval mode while training the predictor."""
        super().train(mode)
        self.encoder.eval()
        if not self.finetune_predictor:
            self.predictor.eval()
        return self

    def forward(
        self,
        context_rgb: torch.Tensor | Sequence[torch.Tensor],
        depth_t: torch.Tensor,
        ego_motions: torch.Tensor | Sequence[torch.Tensor],
        *,
        future_rgb: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> ProjectedPredictorOutput:
        """Run four-step projected latent forecasting for one KITTI sample.

        Parameters
        ----------
        context_rgb:
            Five observed frames ``[I_{t-4}, ..., I_t]``.  Accepted layouts are
            ``(5,H,W,3)``, ``(5,3,H,W)``, or a length-five sequence.
            RGB values should be in ``[0,1]``; uint8 input is also accepted.
        depth_t:
            Metric z-depth for ``I_t``, shape ``(H,W)``.
        ego_motions:
            Four source-relative motions ``t->t+1 ... t->t+4``, each ``(3,)``.
        future_rgb:
            Optional four real future frames.  When supplied they are encoded
            in frozen ViT-B image mode and returned as ``target_latents``.
            Ground-truth future video encoding is intentionally unnecessary for
            the current residual objective.
        """
        context_frames = self._normalise_frame_sequence(
            context_rgb, expected=5, name="context_rgb"
        )

        # 1. Deterministic geometry + staggered video stream construction.
        streams = self.stream_builder.build(
            context_rgb=context_frames,
            depth_t=depth_t,
            ego_motions=ego_motions,
        )

        # 2. Frozen video encoder -> pretrained predictor hidden state for each
        #    staggered stream.  Dense hidden fields include both context and
        #    masked positions because both are allowed to receive corrections.
        hidden_s1 = self._encode_stream_to_dense_predictor_hidden(streams.stream_1)
        hidden_s2 = self._encode_stream_to_dense_predictor_hidden(streams.stream_2)

        # Shapes: (1, 4, Hp*Wp, 384).
        hidden_s1 = self._reshape_temporal_hidden(hidden_s1, streams.stream_1)
        hidden_s2 = self._reshape_temporal_hidden(hidden_s2, streams.stream_2)

        # Temporal alignment of the staggered tubelets:
        #   S2 plane 2 -> t   -> t+1   (V0)
        #   S1 plane 2 -> t+1 -> t+2   (V1)
        #   S2 plane 3 -> t+2 -> t+3   (V2)
        #   S1 plane 3 -> t+3 -> t+4   (V3)
        transport_hidden = torch.stack(
            [
                hidden_s2[:, 2],
                hidden_s1[:, 2],
                hidden_s2[:, 3],
                hidden_s1[:, 3],
            ],
            dim=1,
        )  # (1,4,N,384)

        corrections = self.correction_head(transport_hidden)  # (1,4,N,768)

        # 3. Frozen image-mode encoder constructs the static latent basis:
        #    known Z_t and one latent for each masked RGB warp.
        z_t = self.encode_images([context_frames[-1]])  # (1,N,768)

        warped_frames = [
            self._warp_rgb_for_image_encoder(warp)
            for warp in streams.warps
        ]
        warped_latents = self.encode_images(warped_frames).unsqueeze(0)
        # encode_images returns (4,N,768); retain model batch -> (1,4,N,768).

        patch_valid = torch.stack(
            [
                torch.as_tensor(warp.patch_valid, dtype=torch.bool).reshape(-1)
                for warp in streams.warps
            ],
            dim=0,
        ).to(self.device)
        patch_valid = patch_valid.unsqueeze(0)  # (1,4,N)

        if warped_latents.shape[:3] != corrections.shape[:3]:
            raise RuntimeError(
                "Warp/image and predictor grids disagree: "
                f"warped_latents={tuple(warped_latents.shape)}, "
                f"corrections={tuple(corrections.shape)}."
            )

        # 4. Autoregressive deterministic proposal + learned correction.
        proposals: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []
        z_previous = z_t

        for k in range(self.NUM_FUTURE_STEPS):
            z0 = self.latent_proposal(
                z_previous=z_previous,
                z_warped=warped_latents[:, k],
                patch_valid=patch_valid[:, k],
            )
            z_hat = z0 + corrections[:, k]
            proposals.append(z0)
            predictions.append(z_hat)
            z_previous = z_hat

        proposals_t = torch.stack(proposals, dim=1)      # (1,4,N,768)
        predictions_t = torch.stack(predictions, dim=1)  # (1,4,N,768)

        target_latents = None
        if future_rgb is not None:
            future_frames = self._normalise_frame_sequence(
                future_rgb, expected=4, name="future_rgb"
            )
            target_latents = self.encode_images(future_frames).unsqueeze(0)

        return ProjectedPredictorOutput(
            predictions=predictions_t,
            proposals=proposals_t,
            corrections=corrections,
            transport_hidden=transport_hidden,
            warped_latents=warped_latents,
            patch_valid=patch_valid,
            z_t=z_t,
            target_latents=target_latents,
            streams=streams,
        )

    @torch.no_grad()
    def encode_images(
        self,
        frames: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Encode one or more RGB frames with frozen ViT-B in image mode.

        Returns ``(K, Hp*Wp, 768)``.  The encoder is always evaluated without
        gradients because this representation defines the static target space.
        """
        frame_list = self._normalise_frame_sequence_any(frames, name="frames")
        x = torch.stack([self._to_hwc01(f) for f in frame_list], dim=0)
        x = self._rgb_bhwc_to_vjepa_image(x).to(self.device)
        z = self.encoder(x, masks=None, training=False)
        if z.ndim != 3 or z.shape[-1] != self.image_dim:
            raise RuntimeError(
                "Unexpected image-encoder output shape: "
                f"{tuple(z.shape)}; expected (*,*,{self.image_dim})."
            )
        return z

    # ------------------------------------------------------------------
    # Frozen video encoder + pretrained predictor body
    # ------------------------------------------------------------------

    def _encode_stream_to_dense_predictor_hidden(
        self,
        stream: VideoStream,
    ) -> torch.Tensor:
        """Return the full ordered 384-D predictor hidden grid for one stream.

        This reproduces the released predictor's computation through
        ``predictor_norm`` but deliberately stops before ``predictor_proj`` and
        ``predictor_proj_context``.  Runtime ``T/H/W`` are passed into every
        RoPE block so non-square KITTI grids are decoded correctly.
        """
        x_video = self._rgb_cthw_to_vjepa_video(stream.rgb).to(self.device)
        masks_x = stream.masks_x.to(self.device)
        masks_y = stream.masks_y.to(self.device)

        # Invalid tubelet tokens are deleted before frozen encoder self-attention.
        with torch.no_grad():
            context_tokens = self.encoder(
                x_video,
                masks=masks_x,
                training=False,
            )

        return self._predict_dense_hidden(
            context_tokens=context_tokens,
            context_indices=masks_x,
            target_indices=masks_y,
            grid_thw=tuple(int(v) for v in stream.tubelet_patch_valid.shape),
        )

    def _predict_dense_hidden(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        grid_thw: tuple[int, int, int],
    ) -> torch.Tensor:
        """Run predictor body and scatter context/target hidden states densely."""
        p = self.predictor
        if not getattr(p, "use_rope", False):
            raise NotImplementedError("ProjectedPredictor currently requires RoPE.")
        if getattr(p, "mask_tokens", None) is None or p.num_mask_tokens == 0:
            raise ValueError("Predictor must be constructed with learned mask tokens.")
        if getattr(p, "chop_last_n_tokens", 0) != 0:
            raise NotImplementedError("chop_last_n_tokens is not supported.")

        if context_tokens.ndim != 3:
            raise ValueError("context_tokens must be [B,N_context,D].")
        B = context_tokens.shape[0]
        if B != 1:
            raise ValueError(
                "The initial ProjectedPredictor implementation is per-sample; "
                f"expected B=1, got B={B}."
            )
        if context_indices.shape[0] != B or target_indices.shape[0] != B:
            raise ValueError("Mask batch size must match context token batch size.")

        Tp, Hp, Wp = grid_thw
        total_tokens = Tp * Hp * Wp
        all_indices = torch.cat([context_indices, target_indices], dim=1)
        if all_indices.shape[1] != total_tokens:
            raise ValueError(
                "Context and target masks must partition the complete video token "
                f"grid ({total_tokens} tokens), got {all_indices.shape[1]}."
            )
        if torch.unique(all_indices[0]).numel() != total_tokens:
            raise ValueError("Context and target masks overlap or omit token indices.")
        if all_indices.numel() and (
            int(all_indices.min()) < 0 or int(all_indices.max()) >= total_tokens
        ):
            raise ValueError("A predictor token index falls outside grid_thw.")

        # Released ViT-B: 768 -> 384 pretrained predictor input projection.
        x = p.predictor_embed(context_tokens)
        n_context = x.shape[1]
        if x.shape[-1] != self.predictor_dim:
            raise RuntimeError(
                f"predictor_embed produced {x.shape[-1]} dims, expected "
                f"{self.predictor_dim}."
            )

        # A target location starts from the selected learned 384-D mask token.
        token = p.mask_tokens[self.mask_index % p.num_mask_tokens]
        target_tokens = token.expand(B, target_indices.shape[1], -1)

        # Predictor self-attention operates on one complete sequence, sorted back
        # into original spatiotemporal order for RoPE.
        x = torch.cat([x, target_tokens], dim=1)
        masks = all_indices
        order = torch.argsort(masks, dim=1)
        masks_sorted = torch.gather(masks, 1, order)
        x = torch.gather(
            x,
            1,
            order.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
        )

        if getattr(p, "modality_embedding", False):
            x = x + p.video_mod_embed

        for block in p.predictor_blocks:
            x, _ = block(
                x,
                mask=masks_sorted,
                T=Tp,
                H_patches=Hp,
                W_patches=Wp,
                mode="video",
            )

        x = p.predictor_norm(x)

        # Restore the API's [context, target] concatenation order, then scatter
        # those hidden states into one dense row-major (t,h,w) field.
        inverse_order = torch.argsort(order, dim=1)
        x = torch.gather(
            x,
            1,
            inverse_order.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
        )
        context_hidden = x[:, :n_context]
        target_hidden = x[:, n_context:]
        all_hidden = torch.cat([context_hidden, target_hidden], dim=1)

        dense = all_hidden.new_zeros((B, total_tokens, self.predictor_dim))
        dense = dense.scatter(
            1,
            all_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim),
            all_hidden,
        )
        return dense

    # ------------------------------------------------------------------
    # Shapes / preprocessing
    # ------------------------------------------------------------------

    def _reshape_temporal_hidden(
        self,
        dense_hidden: torch.Tensor,
        stream: VideoStream,
    ) -> torch.Tensor:
        Tp, Hp, Wp = stream.tubelet_patch_valid.shape
        N = Hp * Wp
        expected = Tp * N
        if dense_hidden.shape != (1, expected, self.predictor_dim):
            raise RuntimeError(
                f"Unexpected dense hidden shape {tuple(dense_hidden.shape)}; "
                f"expected {(1, expected, self.predictor_dim)}."
            )
        return dense_hidden.view(1, Tp, N, self.predictor_dim)

    def _rgb_cthw_to_vjepa_video(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert ``(3,T,H,W)`` RGB to normalized ``(1,3,T,H,W)``."""
        rgb = torch.as_tensor(rgb)
        if rgb.ndim != 4 or rgb.shape[0] != 3:
            raise ValueError(f"Expected video RGB (3,T,H,W), got {tuple(rgb.shape)}.")
        x = self._to_float01(rgb).unsqueeze(0)
        mean = x.new_tensor(VJEPA_MEAN).view(1, 3, 1, 1, 1)
        std = x.new_tensor(VJEPA_STD).view(1, 3, 1, 1, 1)
        return (x - mean) / std

    def _rgb_bhwc_to_vjepa_image(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert ``(B,H,W,3)`` RGB to normalized ``(B,3,1,H,W)``."""
        rgb = torch.as_tensor(rgb)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected image RGB (B,H,W,3), got {tuple(rgb.shape)}.")
        x = self._to_float01(rgb).permute(0, 3, 1, 2).unsqueeze(2)
        mean = x.new_tensor(VJEPA_MEAN).view(1, 3, 1, 1, 1)
        std = x.new_tensor(VJEPA_STD).view(1, 3, 1, 1, 1)
        return (x - mean) / std

    @staticmethod
    def _to_float01(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            return x.float() / 255.0
        return x.float()

    @classmethod
    def _to_hwc01(cls, frame: torch.Tensor) -> torch.Tensor:
        frame = torch.as_tensor(frame)
        if frame.ndim != 3:
            raise ValueError(f"Each RGB frame must be rank-3, got {tuple(frame.shape)}.")
        if frame.shape[-1] == 3:
            return cls._to_float01(frame)
        if frame.shape[0] == 3:
            return cls._to_float01(frame.permute(1, 2, 0).contiguous())
        raise ValueError(
            "Could not infer RGB layout; expected (H,W,3) or (3,H,W), got "
            f"{tuple(frame.shape)}."
        )

    @classmethod
    def _normalise_frame_sequence(
        cls,
        frames: torch.Tensor | Sequence[torch.Tensor],
        *,
        expected: int,
        name: str,
    ) -> list[torch.Tensor]:
        out = cls._normalise_frame_sequence_any(frames, name=name)
        if len(out) != expected:
            raise ValueError(f"{name} must contain {expected} frames, got {len(out)}.")
        return out

    @classmethod
    def _normalise_frame_sequence_any(
        cls,
        frames: torch.Tensor | Sequence[torch.Tensor],
        *,
        name: str,
    ) -> list[torch.Tensor]:
        if isinstance(frames, torch.Tensor):
            if frames.ndim == 3:
                seq = [frames]
            elif frames.ndim == 4:
                seq = [frames[i] for i in range(frames.shape[0])]
            else:
                raise ValueError(
                    f"{name} tensor must have rank 3 or 4, got {tuple(frames.shape)}."
                )
        else:
            seq = list(frames)
        if not seq:
            raise ValueError(f"{name} must contain at least one frame.")
        return [cls._to_hwc01(frame) for frame in seq]

    @staticmethod
    def _warp_rgb_for_image_encoder(warp: Any) -> torch.Tensor:
        # Use the whole-patch-masked RGB because patch validity defines which
        # image-mode latent rows are subsequently trusted by LatentProposal.
        rgb = getattr(warp, "rgb_patch_masked", None)
        if rgb is None:
            raise TypeError("Warp output must expose `rgb_patch_masked`.")
        return rgb

    # ------------------------------------------------------------------
    # Model configuration / validation
    # ------------------------------------------------------------------

    def _validate_predictor_interface(self) -> None:
        p = self.predictor
        required = (
            "predictor_embed",
            "mask_tokens",
            "num_mask_tokens",
            "predictor_blocks",
            "predictor_norm",
        )
        missing = [name for name in required if not hasattr(p, name)]
        if missing:
            raise TypeError(
                "Predictor does not expose the expected V-JEPA 2.1 interface: "
                + ", ".join(missing)
            )
        if p.mask_tokens is None or p.num_mask_tokens <= 0:
            raise ValueError("Predictor must have learned mask tokens enabled.")

    def _configure_predictor_gradients(self) -> None:
        # First respect the high-level post-training choice.
        self._set_requires_grad(self.predictor, self.finetune_predictor)

        # The original 384->1664 teacher-space heads are intentionally unused.
        for name in ("predictor_proj", "predictor_proj_context"):
            module = getattr(self.predictor, name, None)
            if module is not None:
                self._set_requires_grad(module, False)

        if not self.finetune_predictor:
            self.predictor.eval()

    @staticmethod
    def _set_requires_grad(module: nn.Module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(value)


__all__ = [
    "ProjectedPredictor",
    "ProjectedPredictorOutput",
]
