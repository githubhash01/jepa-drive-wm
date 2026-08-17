"""Shared train/validation/test DataLoader wiring over a KITTISplit.

Both KITTI data interfaces (dense per-frame and rollout windows) need the same
scaffold: one dataset and one DataLoader per split member, shuffling only the
training loader, plus a check that no sequence appears in two splits.

    train      shuffle=True
    validation shuffle=False
    test       shuffle=False

Shuffling the training split is not cosmetic: samples from one sequence are
near-duplicates of their neighbours (adjacent frames, or windows overlapping in
all but one frame), so an in-order epoch would feed batches of near-identical
samples from one stretch of one sequence. Validation and test stay in order so
their numbers are reproducible across epochs.

Splitting by sequence (never by frame or window) keeps the splits honest --
data within one sequence is far too correlated to be divided between train
and validation.
"""

from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from jepa_drive_wm.data.splits import KITTISplit


class SplitLoaders:
    """Base class: subclasses call _build_datasets() with a dataset factory."""

    def __init__(
        self,
        split: KITTISplit,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool | None = None,
    ):
        self._assert_disjoint(split)

        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Pinned host memory only helps when there is a GPU to copy to.
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    def _build_datasets(self, make_dataset: Callable[[list[int]], Dataset]) -> None:
        """Build the three datasets and loaders; make_dataset(sequence_numbers) -> Dataset."""
        self.train_dataset = make_dataset(list(self.split.train_sequences))
        self.validation_dataset = make_dataset(list(self.split.validation_sequences))
        self.test_dataset = make_dataset(list(self.split.test_sequences))

        self.train = self._make_loader(self.train_dataset, shuffle=True)
        self.validation = self._make_loader(self.validation_dataset, shuffle=False)
        self.test = self._make_loader(self.test_dataset, shuffle=False)

    @staticmethod
    def _assert_disjoint(split: KITTISplit) -> None:
        parts = {
            "train": split.train_sequences,
            "validation": split.validation_sequences,
            "test": split.test_sequences,
        }
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            shared = set(parts[left]) & set(parts[right])
            if shared:
                raise ValueError(
                    f"Sequences {sorted(shared)} appear in both the {left} and {right} "
                    f"splits. Data within one sequence is near-identical, so this leaks."
                )

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Idle worker startup per epoch is pure overhead once workers exist.
            persistent_workers=self.num_workers > 0,
        )
