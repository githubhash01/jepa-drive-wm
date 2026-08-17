"""Immutable train/val/test sequence splits for KITTI odometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KITTISplit:
    name: str
    train_sequences: tuple[int, ...]
    validation_sequences: tuple[int, ...]
    test_sequences: tuple[int, ...]


# Chosen 2026-08-17 via data/split_scripts/split_validation.py: val mirrors test
# (urban + rural + open-road, all speed regimes), ~82/9/9 frame proportions.
SPLIT_V1 = KITTISplit(
    name="v1",
    train_sequences=(0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21),
    validation_sequences=(3, 5, 17),
    test_sequences=(1, 7, 9),
)
