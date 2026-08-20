"""Build the two staggered V-JEPA video streams used by ProjectedPredictor.

The builder owns *temporal assembly and masking*, not geometry itself. Geometry is
provided by a WarpModule instance.

For five observed frames ordered as::

    [I_{t-4}, I_{t-3}, I_{t-2}, I_{t-1}, I_t]

and source-relative warps from I_t::

    [W_{t+1}, W_{t+2}, W_{t+3}, W_{t+4}],

the two 8-frame streams are::

    Stream 1: [I_{t-3}, I_{t-2}, I_{t-1}, I_t,
               W_{t+1}, W_{t+2}, W_{t+3}, W_{t+4}]

    Stream 2: [I_{t-4}, I_{t-3}, I_{t-2}, I_{t-1},
               I_t, W_{t+1}, W_{t+2}, W_{t+3}]

V-JEPA video patchification uses tubelet_size=2.  To prevent one tubelet from
mixing a trustworthy patch in one frame with an untrustworthy patch in the
other, both frames in each future/transition pair share the validity mask of
that pair's *second* frame:

    Stream 1: (W_{t+1}, W_{t+2}) use S_{t+2}
              (W_{t+3}, W_{t+4}) use S_{t+4}

    Stream 2: (I_t, W_{t+1})     use S_{t+1}
              (W_{t+2}, W_{t+3}) use S_{t+3}

where S_k is WarpOutput.patch_valid for W_k.  The first two tubelets of each
stream are fully observed context.

The returned ``masks_x`` and ``masks_y`` are flattened V-JEPA token indices in
(tubelet, row, column) row-major order.  Their union is the complete 4 x Hp x Wp
token grid:

    masks_x = geometrically/context-valid tubelet tokens
    masks_y = missing target tubelet tokens

This class is intentionally per-sample.  Geometry-derived masks have variable
numbers of valid tokens, so batching/padding policy should live one level above
this utility rather than being hidden here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class VideoStream:
    """One V-JEPA-ready 8-frame stream and its token masks.

    Attributes
    ----------
    rgb:
        Tensor of shape ``(3, 8, H, W)`` ready to receive a batch dimension and
        be passed to the V-JEPA video encoder.
    frame_patch_valid:
        Boolean tensor ``(8, Hp, Wp)`` showing which whole spatial patches are
        retained in each RGB frame after conservative pair masking.
    tubelet_patch_valid:
        Boolean tensor ``(4, Hp, Wp)``.  Since both frames of every tubelet use
        the same patch mask, this is the mask that should be applied to V-JEPA
        video tokens after PatchEmbed3D.
    masks_x:
        Long tensor ``(1, N_context)`` containing flattened context-token IDs.
        Shape matches the mask format expected by V-JEPA ``apply_masks`` for a
        single sample.
    masks_y:
        Long tensor ``(1, N_target)`` containing flattened missing/target IDs.
    frame_labels:
        Human-readable labels useful for debugging/visualisation.
    """

    rgb: torch.Tensor
    frame_patch_valid: torch.Tensor
    tubelet_patch_valid: torch.Tensor
    masks_x: torch.Tensor
    masks_y: torch.Tensor
    frame_labels: tuple[str, ...]

    @property
    def num_context_tokens(self) -> int:
        return int(self.masks_x.shape[1])

    @property
    def num_target_tokens(self) -> int:
        return int(self.masks_y.shape[1])


@dataclass(frozen=True)
class StreamPair:
    """Both staggered streams plus the four source-relative warp results."""

    stream_1: VideoStream
    stream_2: VideoStream
    warps: tuple[Any, Any, Any, Any]


class VideoStreamBuilder:
    """Generate warped futures and assemble the two staggered video streams."""

    NUM_CONTEXT_FRAMES = 5
    NUM_FUTURE_WARPS = 4
    NUM_STREAM_FRAMES = 8
    TUBELET_SIZE = 2

    def __init__(self, warp_module: Any):
        # ``Any`` is deliberate: keeping this utility duck-typed avoids pulling
        # PyTorch3D into modules/tests that only exercise stream assembly.
        self.warp_module = warp_module

    def build(
        self,
        context_rgb: torch.Tensor | Sequence[torch.Tensor],
        depth_t: torch.Tensor,
        ego_motions: torch.Tensor | Sequence[torch.Tensor],
    ) -> StreamPair:
        """Build both video streams for one sample.

        Parameters
        ----------
        context_rgb:
            Five observed RGB frames ordered ``[I_{t-4}, ..., I_t]``. Accepted
            layouts are ``(5,H,W,3)``, ``(5,3,H,W)``, or a length-5 sequence of
            ``(H,W,3)`` / ``(3,H,W)`` tensors.
        depth_t:
            Metric depth for the final observed frame, shape ``(H,W)``.
        ego_motions:
            Four source-relative motions ``t -> t+1 ... t -> t+4``.  Each item
            must have shape ``(3,)`` and match ``WarpModule.warp_sequence``.

        Returns
        -------
        StreamPair
            The two V-JEPA-ready streams, masks, and original WarpOutputs.
        """
        context = self._normalise_context_rgb(context_rgb)
        self._validate_context(context, depth_t)

        motions = self._normalise_ego_motions(ego_motions)
        if len(motions) != self.NUM_FUTURE_WARPS:
            raise ValueError(
                f"Expected {self.NUM_FUTURE_WARPS} ego motions "
                f"(t->t+1 ... t->t+4), got {len(motions)}."
            )

        # Geometry is generated once from the final observed RGB-D frame.
        raw_warps = tuple(
            self.warp_module.warp_sequence(
                rgb=context[-1],
                depth=depth_t,
                ego_motions=motions,
            )
        )
        if len(raw_warps) != self.NUM_FUTURE_WARPS:
            raise RuntimeError(
                "WarpModule.warp_sequence returned an unexpected number of warps: "
                f"{len(raw_warps)}."
            )

        warp_rgbs = [self._warp_rgb(w) for w in raw_warps]
        warp_valid = [self._warp_patch_valid(w) for w in raw_warps]

        # WarpModule renders on the device/dtype of its registered intrinsics.
        # Move observed frames to that same representation before stacking the
        # video streams (e.g. CPU DataLoader tensors + GPU warper).
        stream_device = warp_rgbs[0].device
        stream_dtype = warp_rgbs[0].dtype
        context = [
            frame.to(device=stream_device, dtype=stream_dtype)
            for frame in context
        ]

        self._validate_warp_shapes(context, warp_rgbs, warp_valid)

        # ------------------------------------------------------------------
        # Stream 1
        # [I_-3, I_-2, I_-1, I_0, W_1, W_2, W_3, W_4]
        # Future pairs inherit the mask of their second frame.
        # ------------------------------------------------------------------
        s2 = warp_valid[1]
        s4 = warp_valid[3]
        all_valid = torch.ones_like(s2, dtype=torch.bool)

        stream1_frames = [
            context[1],
            context[2],
            context[3],
            context[4],
            self._apply_patch_mask(warp_rgbs[0], s2),
            self._apply_patch_mask(warp_rgbs[1], s2),
            self._apply_patch_mask(warp_rgbs[2], s4),
            self._apply_patch_mask(warp_rgbs[3], s4),
        ]
        stream1_frame_valid = torch.stack(
            [all_valid, all_valid, all_valid, all_valid, s2, s2, s4, s4],
            dim=0,
        )
        stream1 = self._make_stream(
            stream1_frames,
            stream1_frame_valid,
            frame_labels=(
                "I_t-3", "I_t-2", "I_t-1", "I_t",
                "W_t+1", "W_t+2", "W_t+3", "W_t+4",
            ),
        )

        # ------------------------------------------------------------------
        # Stream 2 (one-frame temporal shift)
        # [I_-4, I_-3, I_-2, I_-1, I_0, W_1, W_2, W_3]
        # Transition pair (I_0, W_1) uses S_1; (W_2, W_3) uses S_3.
        # ------------------------------------------------------------------
        s1 = warp_valid[0]
        s3 = warp_valid[2]

        stream2_frames = [
            context[0],
            context[1],
            context[2],
            context[3],
            self._apply_patch_mask(context[4], s1),
            self._apply_patch_mask(warp_rgbs[0], s1),
            self._apply_patch_mask(warp_rgbs[1], s3),
            self._apply_patch_mask(warp_rgbs[2], s3),
        ]
        stream2_frame_valid = torch.stack(
            [all_valid, all_valid, all_valid, all_valid, s1, s1, s3, s3],
            dim=0,
        )
        stream2 = self._make_stream(
            stream2_frames,
            stream2_frame_valid,
            frame_labels=(
                "I_t-4", "I_t-3", "I_t-2", "I_t-1",
                "I_t", "W_t+1", "W_t+2", "W_t+3",
            ),
        )

        return StreamPair(
            stream_1=stream1,
            stream_2=stream2,
            warps=raw_warps,  # useful later for image-mode encoding/proposals
        )

    # ------------------------------------------------------------------
    # Stream construction
    # ------------------------------------------------------------------

    def _make_stream(
        self,
        frames_hwc: Sequence[torch.Tensor],
        frame_patch_valid: torch.Tensor,
        *,
        frame_labels: tuple[str, ...],
    ) -> VideoStream:
        if len(frames_hwc) != self.NUM_STREAM_FRAMES:
            raise ValueError(
                f"Expected {self.NUM_STREAM_FRAMES} frames, got {len(frames_hwc)}."
            )
        if frame_patch_valid.shape[0] != self.NUM_STREAM_FRAMES:
            raise ValueError("frame_patch_valid must contain one mask per frame.")

        # Both frames of a tubelet must have identical masks.  Assert this
        # rather than silently combining them; it catches stream-builder bugs.
        paired = frame_patch_valid.view(
            self.NUM_STREAM_FRAMES // self.TUBELET_SIZE,
            self.TUBELET_SIZE,
            *frame_patch_valid.shape[1:],
        )
        if not torch.equal(paired[:, 0], paired[:, 1]):
            raise RuntimeError("The two frames in each tubelet must share a mask.")
        tubelet_patch_valid = paired[:, 0].contiguous()  # (4, Hp, Wp)

        masks_x, masks_y = self._vjepa_indices(tubelet_patch_valid)

        # Fixed output layout expected by the video encoder once batch is added:
        # (C,T,H,W) -> caller can use stream.rgb.unsqueeze(0).
        rgb = torch.stack(list(frames_hwc), dim=0).permute(3, 0, 1, 2).contiguous()

        return VideoStream(
            rgb=rgb,
            frame_patch_valid=frame_patch_valid,
            tubelet_patch_valid=tubelet_patch_valid,
            masks_x=masks_x,
            masks_y=masks_y,
            frame_labels=frame_labels,
        )

    @staticmethod
    def _vjepa_indices(tubelet_patch_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a ``(Tp,Hp,Wp)`` validity grid to V-JEPA token indices."""
        if tubelet_patch_valid.ndim != 3:
            raise ValueError(
                "tubelet_patch_valid must have shape (T_patches,H_patches,W_patches)."
            )

        flat_valid = tubelet_patch_valid.reshape(-1)
        all_ids = torch.arange(
            flat_valid.numel(),
            device=flat_valid.device,
            dtype=torch.long,
        )
        masks_x = all_ids[flat_valid].unsqueeze(0)
        masks_y = all_ids[~flat_valid].unsqueeze(0)
        return masks_x, masks_y

    def _apply_patch_mask(
        self,
        rgb_hwc: torch.Tensor,
        patch_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Zero whole RGB patches rejected by ``patch_valid``."""
        if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
            raise ValueError(f"Expected RGB shape (H,W,3), got {tuple(rgb_hwc.shape)}.")
        if patch_valid.ndim != 2:
            raise ValueError("patch_valid must have shape (Hp,Wp).")

        H, W = rgb_hwc.shape[:2]
        Hp, Wp = patch_valid.shape
        if H % Hp != 0 or W % Wp != 0:
            raise ValueError(
                f"Patch mask {tuple(patch_valid.shape)} is incompatible with RGB "
                f"shape {(H, W)}."
            )
        patch_h = H // Hp
        patch_w = W // Wp
        if patch_h != patch_w:
            raise ValueError(
                f"Expected square image patches, got ({patch_h},{patch_w})."
            )

        pixel_valid = (
            patch_valid.to(device=rgb_hwc.device, dtype=torch.bool)
            .repeat_interleave(patch_h, dim=0)
            .repeat_interleave(patch_w, dim=1)
        )
        return torch.where(pixel_valid[..., None], rgb_hwc, torch.zeros_like(rgb_hwc))

    # ------------------------------------------------------------------
    # Input normalisation / validation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_hwc(frame: torch.Tensor) -> torch.Tensor:
        frame = torch.as_tensor(frame)
        if frame.ndim != 3:
            raise ValueError(f"Each RGB frame must be rank-3, got {tuple(frame.shape)}.")
        if frame.shape[-1] == 3:
            return frame
        if frame.shape[0] == 3:
            return frame.permute(1, 2, 0).contiguous()
        raise ValueError(
            "Could not infer RGB layout. Expected (H,W,3) or (3,H,W), "
            f"got {tuple(frame.shape)}."
        )

    def _normalise_context_rgb(
        self,
        context_rgb: torch.Tensor | Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if isinstance(context_rgb, torch.Tensor):
            if context_rgb.ndim != 4:
                raise ValueError(
                    "context_rgb tensor must have shape (5,H,W,3) or (5,3,H,W)."
                )
            if context_rgb.shape[0] != self.NUM_CONTEXT_FRAMES:
                raise ValueError(
                    f"Expected {self.NUM_CONTEXT_FRAMES} context frames, "
                    f"got {context_rgb.shape[0]}."
                )
            frames = [context_rgb[i] for i in range(context_rgb.shape[0])]
        else:
            frames = list(context_rgb)
            if len(frames) != self.NUM_CONTEXT_FRAMES:
                raise ValueError(
                    f"Expected {self.NUM_CONTEXT_FRAMES} context frames, got {len(frames)}."
                )
        return [self._to_hwc(f) for f in frames]

    @staticmethod
    def _normalise_ego_motions(
        ego_motions: torch.Tensor | Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if isinstance(ego_motions, torch.Tensor):
            if ego_motions.ndim != 2 or ego_motions.shape[-1] != 3:
                raise ValueError("ego_motions tensor must have shape (K,3).")
            return [ego_motions[i] for i in range(ego_motions.shape[0])]
        motions = [torch.as_tensor(m) for m in ego_motions]
        for motion in motions:
            if motion.shape != (3,):
                raise ValueError(
                    f"Every ego motion must have shape (3,), got {tuple(motion.shape)}."
                )
        return motions

    @staticmethod
    def _validate_context(context: Sequence[torch.Tensor], depth_t: torch.Tensor) -> None:
        shape = context[0].shape
        if any(frame.shape != shape for frame in context):
            raise ValueError("All context RGB frames must have identical shapes.")
        H, W, C = shape
        if C != 3:
            raise ValueError("Context RGB frames must have three channels.")
        if tuple(depth_t.shape) != (H, W):
            raise ValueError(
                f"depth_t shape {tuple(depth_t.shape)} does not match RGB {(H, W)}."
            )

    @staticmethod
    def _warp_rgb(warp: Any) -> torch.Tensor:
        rgb = getattr(warp, "rgb", None)
        if rgb is None:
            raise TypeError("Warp output must expose an `rgb` tensor.")
        return rgb

    @staticmethod
    def _warp_patch_valid(warp: Any) -> torch.Tensor:
        valid = getattr(warp, "patch_valid", None)
        if valid is None:
            raise TypeError("Warp output must expose a `patch_valid` tensor.")
        return valid.to(dtype=torch.bool)

    @staticmethod
    def _validate_warp_shapes(
        context: Sequence[torch.Tensor],
        warp_rgbs: Sequence[torch.Tensor],
        warp_valid: Sequence[torch.Tensor],
    ) -> None:
        H, W, _ = context[0].shape
        valid_shape = warp_valid[0].shape
        for i, rgb in enumerate(warp_rgbs, start=1):
            if tuple(rgb.shape) != (H, W, 3):
                raise ValueError(
                    f"Warp {i} RGB shape {tuple(rgb.shape)} does not match {(H, W, 3)}."
                )
        for i, valid in enumerate(warp_valid, start=1):
            if valid.ndim != 2 or valid.shape != valid_shape:
                raise ValueError(
                    f"Warp {i} patch_valid has inconsistent shape {tuple(valid.shape)}."
                )
