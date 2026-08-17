import numpy as np
import torch
from torch.utils.data import Dataset

from jepa_drive_wm.data.kitti import KITTISequence
from jepa_drive_wm.data.split_loaders import SplitLoaders
from jepa_drive_wm.data.splits import KITTISplit, SPLIT_V1


class KITTIRolloutDataset(Dataset):
    # KITTI odometry is recorded at 10 Hz.
    FRAME_PERIOD = 0.1

    def __init__(
        self,
        sequence_numbers: list[int],
        context_length: int = 4,
        future_length: int = 2,
        frame_stride: int = 5,
    ):
        self.context_length = context_length
        self.future_length = future_length

        # KITTI is captured at 10 Hz, so consecutive frames are only 0.1s apart --
        # too fine a step to be an interesting prediction target. The stride
        # subsamples every window uniformly: step = frame_stride * FRAME_PERIOD,
        # so the default of 5 puts 0.5s between consecutive window frames.
        #
        # `step_seconds` is the single source of truth for the physical step
        # duration. Everything downstream (e.g. the action normalisation in
        # train_wm.py) should read it from here rather than re-deriving it.
        self.frame_stride = frame_stride
        self.step_seconds = frame_stride * self.FRAME_PERIOD

        # Keep each sequence available for lazy loading.
        self.sequences = {
            sequence_nr: KITTISequence(sequence_nr)
            for sequence_nr in sequence_numbers
        }

        # Each entry identifies one valid temporal window.
        self.index: list[tuple[int, int]] = []

        # A window touches context_length + future_length frames spaced frame_stride
        # apart, so it spans this many raw frames from its start index.
        window_span = (context_length + future_length - 1) * frame_stride + 1

        for sequence_nr, sequence in self.sequences.items():
            number_frames = len(sequence)
            number_windows = number_frames - window_span + 1

            # Every start offset is kept, so windows at different phases within a
            # stride still count as distinct samples.
            for start_index in range(max(0, number_windows)):
                self.index.append((sequence_nr, start_index))

    def __len__(self) -> int:
        return len(self.index)

    def _get_ego_motion(
        self,
        sequence: KITTISequence,
        start_frame: int,
        end_frame: int,
    ) -> torch.Tensor:
        # Supports either name while your KITTISequence code is evolving.
        if hasattr(sequence, "get_ego_motion"):
            motion = sequence.get_ego_motion(start_frame, end_frame)
        else:
            motion = sequence.get_motion(start_frame, end_frame)

        return torch.as_tensor(motion, dtype=torch.float32)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sequence_nr, start_index = self.index[item]
        sequence = self.sequences[sequence_nr]

        # Frames are spaced frame_stride apart, so a window starting at start_index
        # touches start, start + stride, start + 2 * stride, ... The first
        # context_length of those are context; the rest are the targets.
        stride = self.frame_stride
        window_frames = [
            start_index + step * stride
            for step in range(self.context_length + self.future_length)
        ]
        context_frames = window_frames[: self.context_length]
        future_frames = window_frames[self.context_length :]

        context_latents = [
            torch.from_numpy(
                sequence.get_vjepa_features(frame_index, dtype=np.float16)
            ).float()
            for frame_index in context_frames
        ]

        future_latents = [
            torch.from_numpy(
                sequence.get_vjepa_features(frame_index, dtype=np.float16)
            ).float()
            for frame_index in future_frames
        ]

        # Ego motion accumulated over each stride-long step:
        # Motion 0: last context frame -> first future frame
        # Motion 1: first future frame -> second future frame
        future_ego_motions = [
            self._get_ego_motion(sequence, from_frame, to_frame)
            for from_frame, to_frame in zip(
                [context_frames[-1]] + future_frames[:-1], future_frames
            )
        ]

        return {
            "context_latents": torch.stack(context_latents),
            "future_ego_motions": torch.stack(future_ego_motions),
            "future_latents": torch.stack(future_latents),
            "sequence_nr": torch.tensor(sequence_nr, dtype=torch.long),
            "start_index": torch.tensor(start_index, dtype=torch.long),
        }


class KITTIRolloutLoaders(SplitLoaders):
    """
    Train / validation / test DataLoaders over KITTIRolloutDataset, built from
    a KITTISplit and sharing one window configuration (shuffle/leak-check
    scaffold in data/split_loaders.py).

    `step_seconds` (the physical duration of one prediction step) is computed once
    by the datasets and re-exported here. Consumers should take it from this
    attribute -- never recompute it from frame_stride.
    """

    def __init__(
        self,
        split: KITTISplit = SPLIT_V1,
        context_length: int = 4,
        future_length: int = 2,
        frame_stride: int = 5,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool | None = None,
    ):
        super().__init__(split, batch_size, num_workers, pin_memory)

        dataset_kwargs = dict(
            context_length=context_length,
            future_length=future_length,
            frame_stride=frame_stride,
        )
        self._build_datasets(lambda seqs: KITTIRolloutDataset(seqs, **dataset_kwargs))

        # All three datasets share one window configuration, so one step duration.
        self.step_seconds = self.train_dataset.step_seconds

    def __repr__(self) -> str:
        return (
            f"KITTIRolloutLoaders(split={self.split.name}, step={self.step_seconds}s, "
            f"batch_size={self.batch_size}, "
            f"windows: train={len(self.train_dataset)}, "
            f"validation={len(self.validation_dataset)}, test={len(self.test_dataset)})"
        )


def main() -> None:
    loaders = KITTIRolloutLoaders(
        split=SPLIT_V1,
        batch_size=1,
        num_workers=0,  # Keep at zero while debugging.
    )
    print(loaders)

    batch = next(iter(loaders.train))

    print("\nBatch shapes:")
    print("context_latents:   ", batch["context_latents"].shape)
    print("future_ego_motions:", batch["future_ego_motions"].shape)
    print("future_latents:    ", batch["future_latents"].shape)
    print("sequence_nr:       ", batch["sequence_nr"])
    print("start_index:       ", batch["start_index"])

    # Expected:
    # context_latents:    [B, 4, N, D]
    # future_ego_motions: [B, 2, 3]     -- [forward, right, yaw_right] per future step
    # future_latents:     [B, 2, N, D]


if __name__ == "__main__":
    main()
