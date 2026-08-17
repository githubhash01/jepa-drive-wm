"""Evaluate a KITTI train/val/test split against the learned profiles.

By default this evaluates SPLIT_V1 from data/splits.py (test [1, 7, 9],
validation [3, 5, 17], train = the rest); point SPLIT at another KITTISplit
to evaluate a different proposal.

For each split this reports:
  1) frame / duration proportion,
  2) pooled speed-regime composition (stationary / slow / medium / fast,
     using the dataset-wide bins learned by design_kitti_splits),
  3) turn profile (turn episodes, turns per minute, strong-turn window
     fraction, background yaw variability),
  4) environment composition (urban / rural / open-road / mixed),
and finally flags regimes or environments that appear in val/test but are
missing (or nearly missing) in train.
"""

from __future__ import annotations

import numpy as np

from jepa_drive_wm.data.kitti import KITTISequence
from jepa_drive_wm.data.split_scripts.design_kitti_splits import (
    DatasetBinner,
    DrivingProfile,
    DrivingProfileProcessor,
    LearnedBins,
    SPEED_LABELS,
    SequenceMeasurements,
    print_profiles,
)
from jepa_drive_wm.data.splits import SPLIT_V1

SPLIT = SPLIT_V1
ALL_SEQUENCES = sorted(
    SPLIT.train_sequences + SPLIT.validation_sequences + SPLIT.test_sequences
)

ENVIRONMENT_LABELS = ("urban", "rural", "open-road", "mixed")


def build_splits() -> dict[str, list[int]]:
    return {
        "train": list(SPLIT.train_sequences),
        "val": list(SPLIT.validation_sequences),
        "test": list(SPLIT.test_sequences),
    }


def speed_regime_fractions(
    bins: LearnedBins, measurements: list[SequenceMeasurements]
) -> np.ndarray:
    """Pooled (stationary, slow, medium, fast) window fractions over sequences."""
    speed = np.concatenate([m.speed_mps for m in measurements])
    labels = np.zeros(speed.shape, dtype=np.int64)
    moving = speed > bins.stationary_max_mps
    labels[moving] = 1 + np.digitize(speed[moving], bins.moving_speed_edges_mps)
    fractions = np.bincount(labels, minlength=4).astype(np.float64)
    return fractions / fractions.sum()


def report_split(
    name: str,
    seq_nrs: list[int],
    bins: LearnedBins,
    measurements: dict[int, SequenceMeasurements],
    profiles: dict[int, DrivingProfile],
    frames: dict[int, int],
) -> dict:
    ms = [measurements[s] for s in seq_nrs]
    ps = [profiles[s] for s in seq_nrs]

    n_frames = sum(frames[s] for s in seq_nrs)
    duration = sum(m.duration_s for m in ms)

    speed_fracs = speed_regime_fractions(bins, ms)

    yaw = np.concatenate([m.yaw_rate_rad_s for m in ms])
    strong_frac = float(np.mean(np.abs(yaw) >= bins.strong_turn_threshold_rad_s))
    turns = sum(p.turn_count for p in ps)

    env_counts = {
        env: sum(p.environment_class == env for p in ps) for env in ENVIRONMENT_LABELS
    }

    print(f"\n=== {name}: sequences {[f'{s:02d}' for s in seq_nrs]} ===")
    print_profiles(ps)
    print(f"\nframes:            {n_frames}")
    print(f"duration:          {duration / 60.0:.1f} min")
    print("speed regimes:     "
          + "  ".join(f"{lbl} {f:.0%}" for lbl, f in zip(SPEED_LABELS, speed_fracs)))
    print(f"turn episodes:     {turns}  ({turns / (duration / 60.0):.1f} per min)")
    print(f"strong-turn windows: {strong_frac:.1%}")
    print("environments:      "
          + "  ".join(f"{env} {n}" for env, n in env_counts.items() if n))

    return {
        "n_frames": n_frames,
        "duration_s": duration,
        "speed_fracs": speed_fracs,
        "turns_per_min": turns / (duration / 60.0),
        "strong_turn_frac": strong_frac,
        "env_counts": env_counts,
    }


def report_coverage(stats: dict[str, dict]) -> None:
    """Flag regimes/environments held-out splits rely on but train barely has."""
    print("\n=== coverage checks (val/test vs train) ===")
    issues = 0
    train = stats["train"]
    for name in ("val", "test"):
        s = stats[name]
        for k, lbl in enumerate(SPEED_LABELS):
            if s["speed_fracs"][k] > 0.05 and train["speed_fracs"][k] < 0.01:
                print(f"  WARNING: {name} is {s['speed_fracs'][k]:.0%} '{lbl}' "
                      f"but train has only {train['speed_fracs'][k]:.1%}")
                issues += 1
        for env, n in s["env_counts"].items():
            if n and not train["env_counts"][env]:
                print(f"  WARNING: {name} contains '{env}' sequences but train has none")
                issues += 1
    if not issues:
        print("  none — every regime/environment used in val/test is represented in train")


def main() -> None:
    splits = build_splits()
    print("Proposed split")
    for name, seq_nrs in splits.items():
        print(f"  {name:<6} {[f'{s:02d}' for s in seq_nrs]}")

    binner = DatasetBinner(
        ALL_SEQUENCES,
        motion_interval_s=0.5,
        semantic_sample_s=5.0,
        stationary_max_mps=0.5,
    )
    bins = binner.fit()
    processor = DrivingProfileProcessor(bins, binner.measurements, turn_gap_s=0.5)
    profiles = {p.sequence_nr: p for p in processor.build_all()}
    frames = {s: len(KITTISequence(sequence_nr=s)) for s in ALL_SEQUENCES}

    total_frames = sum(frames.values())
    total_duration = sum(m.duration_s for m in binner.measurements.values())

    stats = {
        name: report_split(name, seq_nrs, bins, binner.measurements, profiles, frames)
        for name, seq_nrs in splits.items()
    }

    print("\n=== proportions ===")
    for name, s in stats.items():
        print(f"  {name:<6} {s['n_frames'] / total_frames:5.1%} of frames "
              f"({s['n_frames']}), {s['duration_s'] / total_duration:5.1%} of duration "
              f"({s['duration_s'] / 60.0:.1f} min)")

    report_coverage(stats)


if __name__ == "__main__":
    main()
