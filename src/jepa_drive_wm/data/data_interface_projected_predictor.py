"""Data interface for the geometry-guided ProjectedPredictor.

This dataset intentionally exposes *raw facts* needed by the projected-predictor
pipeline rather than cached V-JEPA latents.

For each sample it returns five observed frames and four future target frames,
all separated by ``frame_stride`` raw KITTI frames::

    context: [I_{t-4}, I_{t-3}, I_{t-2}, I_{t-1}, I_t]
    future:  [I_{t+1}, I_{t+2}, I_{t+3}, I_{t+4}]

The deterministic RGB warp is always constructed from the final observed frame
``I_t`` and its depth map, so the ego motions returned here are *source-relative*::

    T(t -> t+1), T(t -> t+2), T(t -> t+3), T(t -> t+4)

rather than incremental motions between adjacent future frames.

The default image size is 384 x 1248, matching the current V-JEPA / KITTI
pipeline. RGB and depth are resized in the dataset and the camera intrinsics are
scaled consistently.

Current ``ProjectedPredictor.forward`` is deliberately per-sample because the
geometry-derived token masks have variable lengths. ``projected_predictor_inputs``
therefore converts a batch-size-one DataLoader batch directly into keyword
arguments for the model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from jepa_drive_wm.data.kitti import KITTISequence
from jepa_drive_wm.data.split_loaders import SplitLoaders
from jepa_drive_wm.data.splits import KITTISplit, SPLIT_V1


class KITTIProjectedPredictorDataset(Dataset):
    """Temporal windows containing everything needed by ProjectedPredictor.

    Parameters
    ----------
    sequence_numbers:
        KITTI odometry sequence IDs included in this dataset.
    context_length:
        Number of known RGB frames. The current projected-predictor design uses
        exactly five because the two staggered V-JEPA streams need
        ``I_{t-4}, ..., I_t``.
    future_length:
        Number of future RGB targets / geometric warps. The current design uses
        exactly four.
    frame_stride:
        Number of raw KITTI frames between consecutive model timesteps. KITTI is
        10 Hz, so the default of 5 corresponds to 0.5 s.
    image_size:
        ``(H, W)`` returned to the model. RGB uses bilinear interpolation and
        depth uses nearest-neighbour interpolation. Intrinsics are scaled to the
        resized image coordinates.
    """

    FRAME_PERIOD = 0.1  # KITTI odometry is 10 Hz.

    def __init__(
        self,
        sequence_numbers: list[int],
        *,
        context_length: int = 5,
        future_length: int = 4,
        frame_stride: int = 5,
        image_size: tuple[int, int] = (384, 1248),
    ) -> None:
        super().__init__()

        if context_length != 5:
            raise ValueError(
                "ProjectedPredictor currently requires context_length=5 "
                "for its two staggered eight-frame video streams."
            )
        if future_length != 4:
            raise ValueError(
                "ProjectedPredictor currently requires future_length=4."
            )
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive.")

        H, W = image_size
        if H <= 0 or W <= 0:
            raise ValueError(f"Invalid image_size={image_size}.")
        if H % 16 != 0 or W % 16 != 0:
            raise ValueError(
                "ProjectedPredictor currently assumes image dimensions divisible "
                "by the V-JEPA patch size 16."
            )

        self.context_length = int(context_length)
        self.future_length = int(future_length)
        self.frame_stride = int(frame_stride)
        self.step_seconds = self.frame_stride * self.FRAME_PERIOD
        self.image_size = (int(H), int(W))

        # Keep sequence objects for lazy image/depth loading.
        self.sequences = {
            sequence_nr: KITTISequence(sequence_nr)
            for sequence_nr in sequence_numbers
        }

        # Native (H, W) per sequence, filled lazily on first intrinsics scaling
        # so calibration does not require decoding an extra image per sample.
        self._native_sizes: dict[int, tuple[int, int]] = {}

        # Each entry is (sequence number, raw KITTI start-frame index).
        self.index: list[tuple[int, int]] = []

        total_frames = self.context_length + self.future_length
        window_span = (total_frames - 1) * self.frame_stride + 1

        for sequence_nr, sequence in self.sequences.items():
            number_windows = len(sequence) - window_span + 1
            for start_index in range(max(0, number_windows)):
                self.index.append((sequence_nr, start_index))

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------
    # Public sample construction
    # ------------------------------------------------------------------

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sequence_nr, start_index = self.index[item]
        sequence = self.sequences[sequence_nr]

        sampled_frames = [
            start_index + step * self.frame_stride
            for step in range(self.context_length + self.future_length)
        ]
        context_frame_ids = sampled_frames[: self.context_length]
        future_frame_ids = sampled_frames[self.context_length :]
        source_frame_id = context_frame_ids[-1]

        # RGB is returned in standard torch video/image layout [T, C, H, W].
        context_rgb = torch.stack(
            [self._load_rgb(sequence, frame_id) for frame_id in context_frame_ids],
            dim=0,
        )
        future_rgb = torch.stack(
            [self._load_rgb(sequence, frame_id) for frame_id in future_frame_ids],
            dim=0,
        )

        # Geometry is reconstructed only from the final known observation I_t.
        depth_t = self._load_depth(sequence, source_frame_id)

        # WarpModule.warp_sequence expects every motion relative to the same
        # source frame I_t, not future-to-future incremental motion.
        ego_motions = torch.stack(
            [
                self._get_ego_motion(sequence, source_frame_id, future_frame_id)
                for future_frame_id in future_frame_ids
            ],
            dim=0,
        )  # [4, 3]

        intrinsics = self._scaled_intrinsics(sequence_nr, sequence, source_frame_id)

        return {
            # Direct inputs to ProjectedPredictor.forward().
            "context_rgb": context_rgb,          # [5, 3, H, W]
            "depth_t": depth_t,                  # [H, W]
            "ego_motions": ego_motions,          # [4, 3], all source-relative
            "future_rgb": future_rgb,            # [4, 3, H, W], train/eval target

            # Geometry/debug metadata. The current model instance owns its
            # WarpModule/K, but exposing K here makes the per-sample calibration
            # explicit and enables a later dynamic-intrinsics warper.
            "intrinsics": intrinsics,            # [3, 3], scaled to H x W
            "context_frame_ids": torch.tensor(context_frame_ids, dtype=torch.long),
            "future_frame_ids": torch.tensor(future_frame_ids, dtype=torch.long),
            "source_frame_id": torch.tensor(source_frame_id, dtype=torch.long),
            "sequence_nr": torch.tensor(sequence_nr, dtype=torch.long),
            "start_index": torch.tensor(start_index, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # KITTI loading / resizing
    # ------------------------------------------------------------------

    def _load_rgb(self, sequence: KITTISequence, frame_id: int) -> torch.Tensor:
        """Load one RGB frame as float32 [3,H,W] in [0,1]."""
        image = np.asarray(sequence.get_image(frame_id))
        rgb = torch.as_tensor(image)

        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(
                f"Expected KITTI RGB [H,W,3], got {tuple(rgb.shape)} "
                f"for frame {frame_id}."
            )

        # KITTISequence in the current project normally returns float RGB in
        # [0,1], but supporting uint8 here keeps the dataset interface robust.
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().div_(255.0)
        else:
            rgb = rgb.float()
            if rgb.numel() and float(rgb.max()) > 1.5:
                rgb = rgb / 255.0

        rgb = rgb.permute(2, 0, 1).contiguous()  # [3,H,W]
        if tuple(rgb.shape[-2:]) != self.image_size:
            rgb = F.interpolate(
                rgb.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )[0]

        return rgb

    def _load_depth(self, sequence: KITTISequence, frame_id: int) -> torch.Tensor:
        """Load metric z-depth as float32 [H,W]."""
        depth = torch.as_tensor(
            np.asarray(sequence.get_depth(frame_id)), dtype=torch.float32
        )
        if depth.ndim != 2:
            raise ValueError(
                f"Expected KITTI depth [H,W], got {tuple(depth.shape)} "
                f"for frame {frame_id}."
            )

        if tuple(depth.shape) != self.image_size:
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=self.image_size,
                mode="nearest",
            )[0, 0]

        return depth

    def _get_ego_motion(
        self,
        sequence: KITTISequence,
        start_frame: int,
        end_frame: int,
    ) -> torch.Tensor:
        """Return [forward,right,yaw_right] from start -> end as float32."""
        if hasattr(sequence, "get_ego_motion"):
            motion = sequence.get_ego_motion(start_frame, end_frame)
        else:
            motion = sequence.get_motion(start_frame, end_frame)

        motion = torch.as_tensor(motion, dtype=torch.float32)
        if motion.shape != (3,):
            raise ValueError(
                "Expected ego motion [forward,right,yaw_right] with shape (3,), "
                f"got {tuple(motion.shape)} for {start_frame}->{end_frame}."
            )
        return motion

    def _scaled_intrinsics(
        self,
        sequence_nr: int,
        sequence: KITTISequence,
        frame_id: int,
    ) -> torch.Tensor:
        """Scale K2 from native KITTI resolution to ``self.image_size``."""
        K = torch.as_tensor(sequence.calib.K2, dtype=torch.float32).clone()
        if K.shape != (3, 3):
            raise ValueError(f"Expected calib.K2 shape (3,3), got {tuple(K.shape)}.")

        # Native dimensions are read from a real frame once per sequence (they
        # can differ between KITTI sequences but not within one) and cached so
        # later samples avoid an extra image decode.
        if sequence_nr not in self._native_sizes:
            native = np.asarray(sequence.get_image(frame_id))
            self._native_sizes[sequence_nr] = tuple(native.shape[:2])
        orig_H, orig_W = self._native_sizes[sequence_nr]
        H, W = self.image_size

        K[0, :] *= W / orig_W
        K[1, :] *= H / orig_H
        return K


class KITTIProjectedPredictorLoaders(SplitLoaders):
    """Train/validation/test DataLoaders for ProjectedPredictor.

    The current ProjectedPredictor implementation is intentionally per-sample
    because its geometry-derived ``masks_x`` / ``masks_y`` have variable token
    counts. Keep ``batch_size=1`` until batching/padding is implemented at the
    model level.
    """

    def __init__(
        self,
        split: KITTISplit = SPLIT_V1,
        *,
        frame_stride: int = 5,
        image_size: tuple[int, int] = (384, 1248),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool | None = None,
    ) -> None:
        if batch_size != 1:
            raise ValueError(
                "ProjectedPredictor currently supports batch_size=1 because "
                "geometry-derived predictor masks have variable lengths."
            )

        super().__init__(split, batch_size, num_workers, pin_memory)

        dataset_kwargs = dict(
            context_length=5,
            future_length=4,
            frame_stride=frame_stride,
            image_size=image_size,
        )
        self._build_datasets(
            lambda seqs: KITTIProjectedPredictorDataset(seqs, **dataset_kwargs)
        )

        self.step_seconds = self.train_dataset.step_seconds
        self.image_size = image_size

    def __repr__(self) -> str:
        return (
            "KITTIProjectedPredictorLoaders("
            f"split={self.split.name}, step={self.step_seconds}s, "
            f"image_size={self.image_size}, batch_size={self.batch_size}, "
            f"windows: train={len(self.train_dataset)}, "
            f"validation={len(self.validation_dataset)}, "
            f"test={len(self.test_dataset)})"
        )


def projected_predictor_inputs(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert a batch-size-one DataLoader batch into model keyword arguments.

    Example
    -------
    ``output = model(**projected_predictor_inputs(batch))``

    This helper intentionally leaves calibration/metadata out of the model call;
    they remain available in ``batch`` for constructing/checking the WarpModule.
    """
    required = ("context_rgb", "depth_t", "ego_motions", "future_rgb")
    for key in required:
        if key not in batch:
            raise KeyError(f"Projected-predictor batch is missing required key {key!r}.")
        if batch[key].shape[0] != 1:
            raise ValueError(
                f"Expected batch_size=1 for {key!r}, got shape {tuple(batch[key].shape)}."
            )

    return {
        "context_rgb": batch["context_rgb"][0],
        "depth_t": batch["depth_t"][0],
        "ego_motions": batch["ego_motions"][0],
        "future_rgb": batch["future_rgb"][0],
    }


def main() -> None:
    loaders = KITTIProjectedPredictorLoaders(
        split=SPLIT_V1,
        batch_size=1,
        num_workers=0,
    )
    print(loaders)

    batch = next(iter(loaders.train))

    print("\nBatch shapes:")
    print("context_rgb:      ", batch["context_rgb"].shape)
    print("future_rgb:       ", batch["future_rgb"].shape)
    print("depth_t:          ", batch["depth_t"].shape)
    print("ego_motions:      ", batch["ego_motions"].shape)
    print("intrinsics:       ", batch["intrinsics"].shape)
    print("context_frame_ids:", batch["context_frame_ids"])
    print("future_frame_ids: ", batch["future_frame_ids"])
    print("sequence_nr:      ", batch["sequence_nr"])

    # Expected DataLoader shapes:
    # context_rgb: [1, 5, 3, 384, 1248]
    # future_rgb:  [1, 4, 3, 384, 1248]
    # depth_t:     [1, 384, 1248]
    # ego_motions: [1, 4, 3]
    # intrinsics:  [1, 3, 3]

    model_kwargs = projected_predictor_inputs(batch)
    print("\nProjectedPredictor.forward kwargs:")
    for key, value in model_kwargs.items():
        print(f"{key:12s}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
