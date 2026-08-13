import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jepa_drive_wm.data.kitti import KITTISequence


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


class KITTIRolloutLoaders:
    """
    The train / validation / test DataLoaders over KITTIRolloutDataset, built from a
    split of KITTI sequence numbers and sharing one window configuration.

        train      shuffle=True
        validation shuffle=False
        test       shuffle=False

    Shuffling the training split is not cosmetic here: window start indices step by a
    single raw frame, so neighbouring windows overlap in all but one of their frames.
    Read in order, a batch would be a handful of near-duplicates from one stretch of
    one sequence. Validation and test stay in order so their numbers are reproducible
    across epochs.

    Splitting by sequence (rather than by window) keeps the splits honest -- windows
    from one sequence overlap far too much to be safely divided between train and
    validation.

    `step_seconds` (the physical duration of one prediction step) is computed once
    by the datasets and re-exported here. Consumers should take it from this
    attribute -- never recompute it from frame_stride.
    """

    def __init__(
        self,
        training_sequences: list[int],
        validation_sequences: list[int],
        test_sequences: list[int],
        context_length: int = 4,
        future_length: int = 2,
        frame_stride: int = 5,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool | None = None,
    ):
        splits = {
            "train": training_sequences,
            "validation": validation_sequences,
            "test": test_sequences,
        }
        self._assert_disjoint(splits)

        self.batch_size = batch_size
        self.num_workers = num_workers
        # Pinned host memory only helps when there is a GPU to copy to.
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

        dataset_kwargs = dict(
            context_length=context_length,
            future_length=future_length,
            frame_stride=frame_stride,
        )
        self.train_dataset = KITTIRolloutDataset(training_sequences, **dataset_kwargs)
        self.validation_dataset = KITTIRolloutDataset(validation_sequences, **dataset_kwargs)
        self.test_dataset = KITTIRolloutDataset(test_sequences, **dataset_kwargs)

        # All three datasets share one window configuration, so one step duration.
        self.step_seconds = self.train_dataset.step_seconds

        self.train = self._make_loader(self.train_dataset, shuffle=True)
        self.validation = self._make_loader(self.validation_dataset, shuffle=False)
        self.test = self._make_loader(self.test_dataset, shuffle=False)

    @staticmethod
    def _assert_disjoint(splits: dict[str, list[int]]) -> None:
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            shared = set(splits[left]) & set(splits[right])
            if shared:
                raise ValueError(
                    f"Sequences {sorted(shared)} appear in both the {left} and {right} "
                    f"splits. Windows within a sequence overlap heavily, so this leaks."
                )

    def _make_loader(self, dataset: KITTIRolloutDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Idle worker startup per epoch is pure overhead once workers exist.
            persistent_workers=self.num_workers > 0,
        )

    def __repr__(self) -> str:
        return (
            f"KITTIRolloutLoaders(step={self.step_seconds}s, batch_size={self.batch_size}, "
            f"windows: train={len(self.train_dataset)}, "
            f"validation={len(self.validation_dataset)}, test={len(self.test_dataset)})"
        )


def main() -> None:
    loaders = KITTIRolloutLoaders(
        training_sequences=[0, 1, 2, 3, 5, 6, 8],
        validation_sequences=[7, 10],
        test_sequences=[4, 9],
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