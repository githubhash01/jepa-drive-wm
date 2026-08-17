import pathlib

import torch
from torch.utils.data import Dataset
from jepa_drive_wm.data.kitti import KITTISequence, SEMANTICS_DIRNAME
from jepa_drive_wm.data.split_loaders import SplitLoaders
from jepa_drive_wm.data.splits import KITTISplit, SPLIT_V1
import numpy as np


"""
Dense-decoding data interface.

Unlike the world model, the dense decoder is not rollout dependent: each sample is
a single frame. The input is that frame's V-JEPA 2.1 ViT-B token grid, laid out as
[C, H, W] for the DPT head; the target is the per-pixel pseudolabel (metric depth
from FoundationStereo, or the OneFormer semantic id map).

Splits are by sequence, never by frame -- neighbouring frames in one sequence are
near-identical, so a frame-level split would leak between train and validation.
Which sequences belong to which split is defined once in data/splits.py.
"""


class _KITTIDenseDataset(Dataset):
    """
    Shared per-frame machinery. Subclasses only say how to read the target.

    Each item:
        features:     [C, H, W]  V-JEPA grid (float32), ready for the DPT head
        target:       [H, W]     depth (float32) or semantic ids (long)
        sequence_nr:  scalar     which sequence this frame came from
        frame_index:  scalar     frame index within that sequence
    """

    def __init__(self, sequence_numbers: list[int]):
        # Keep each sequence available for lazy loading.
        self.sequences = {
            sequence_nr: KITTISequence(sequence_nr)
            for sequence_nr in sequence_numbers
        }

        # Flat (sequence_nr, frame_index) index, skipping frames whose target
        # pseudolabel is missing or corrupt. FoundationStereo leaves gaps (e.g.
        # seq 08 has 71 frames with no depth) and a handful of 0-byte PNGs (seq
        # 00 has 12); indexing them blindly would crash training mid-run. A valid
        # depth/semantic PNG is tens of KB, so "exists and size > 0" is a safe,
        # cheap filter -- no need to open every file.
        self.index: list[tuple[int, int]] = []
        for sequence_nr, sequence in self.sequences.items():
            kept = 0
            for frame_index in range(len(sequence)):
                path = self._target_path(sequence, frame_index)
                if path.exists() and path.stat().st_size > 0:
                    self.index.append((sequence_nr, frame_index))
                    kept += 1
            skipped = len(sequence) - kept
            note = f"  ({skipped} skipped: missing/empty target)" if skipped else ""
            print(f"[{type(self).__name__}] seq {sequence_nr:02d}: {kept}/{len(sequence)} usable{note}")

    def __len__(self) -> int:
        return len(self.index)

    def _target_path(self, sequence: KITTISequence, frame_index: int) -> pathlib.Path:
        """Path to the target pseudolabel for one frame (used to skip gaps)."""
        raise NotImplementedError

    def _load_target(self, sequence: KITTISequence, frame_index: int) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sequence_nr, frame_index = self.index[item]
        sequence = self.sequences[sequence_nr]

        # (grid_h, grid_w, embed_dim) -> [C, H, W] as the DPT head expects.
        grid = sequence.get_vjepa_features(frame_index, as_grid=True, dtype=np.float16)
        features = torch.from_numpy(grid).float().permute(2, 0, 1).contiguous()

        return {
            "features": features,
            "target": self._load_target(sequence, frame_index),
            "sequence_nr": torch.tensor(sequence_nr, dtype=torch.long),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
        }


class KITTIDepthDataset(_KITTIDenseDataset):
    def _target_path(self, sequence: KITTISequence, frame_index: int) -> pathlib.Path:
        return pathlib.Path(sequence.depths[frame_index])

    def _load_target(self, sequence: KITTISequence, frame_index: int) -> torch.Tensor:
        depth = sequence.get_depth(frame_index)          # (H, W) metres
        return torch.from_numpy(np.ascontiguousarray(depth)).float()


class KITTISemanticDataset(_KITTIDenseDataset):
    def _target_path(self, sequence: KITTISequence, frame_index: int) -> pathlib.Path:
        return sequence.sequence_folder / SEMANTICS_DIRNAME / f"{frame_index:06d}.png"

    def _load_target(self, sequence: KITTISequence, frame_index: int) -> torch.Tensor:
        semantics = sequence.get_semantics(frame_index)  # (H, W) class ids
        return torch.from_numpy(np.ascontiguousarray(semantics)).long()


# Task name -> dataset class. Keeps the loader wiring identical for both heads.
DENSE_DATASETS = {
    "depth": KITTIDepthDataset,
    "semantics": KITTISemanticDataset,
}


class KITTIDenseLoaders(SplitLoaders):
    """
    Train / validation / test DataLoaders over a per-frame dense dataset, built
    from a KITTISplit (shuffle/leak-check scaffold in data/split_loaders.py).

    Note: KITTI's native image resolution differs slightly across sequences, so the
    per-pixel targets are not all the same shape. The V-JEPA grid features are, so
    batch_size > 1 is safe only within a single-resolution split; keep batch_size=1
    when a split mixes sequences of different resolutions.
    """

    def __init__(
        self,
        task: str,
        split: KITTISplit = SPLIT_V1,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool | None = None,
    ):
        if task not in DENSE_DATASETS:
            raise ValueError(f"Unknown task {task!r}; expected one of {sorted(DENSE_DATASETS)}.")
        super().__init__(split, batch_size, num_workers, pin_memory)

        self.task = task
        self._build_datasets(DENSE_DATASETS[task])

    def __repr__(self) -> str:
        return (
            f"KITTIDenseLoaders(task={self.task}, split={self.split.name}, "
            f"batch_size={self.batch_size}, "
            f"frames: train={len(self.train_dataset)}, "
            f"validation={len(self.validation_dataset)}, test={len(self.test_dataset)})"
        )


def main() -> None:
    for task in ("depth", "semantics"):
        loaders = KITTIDenseLoaders(
            task=task,
            split=SPLIT_V1,
            batch_size=1,
            num_workers=0,  # Keep at zero while debugging.
        )
        print(loaders)

        batch = next(iter(loaders.train))
        print(f"  features:    {tuple(batch['features'].shape)}  {batch['features'].dtype}")
        print(f"  target:      {tuple(batch['target'].shape)}  {batch['target'].dtype}")
        print(f"  sequence_nr: {batch['sequence_nr'].tolist()}  frame_index: {batch['frame_index'].tolist()}")


if __name__ == "__main__":
    main()
