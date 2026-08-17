"""Visualise the compact KITTI driving profiles produced by design_kitti_splits.py.

This module deliberately contains no profiling or binning logic.  It imports the
existing DatasetBinner and DrivingProfileProcessor, builds the same profiles, and
turns them into a small set of interpretable Matplotlib figures.

Figures
-------
1. speed_regimes.png
   Stacked percentage of stationary / slow / medium / fast motion per sequence.

2. turn_counts.png
   Number of detected strong-turn episodes per sequence.

3. yaw_variability.png
   Background signed-yaw variability after strong turns are removed.

4. environments.png
   Mean building occupancy versus vegetation occupancy, with the learned
   environment boundaries and sequence numbers shown directly on the plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from jepa_drive_wm.data.split_scripts.design_kitti_splits import (
    DatasetBinner,
    DrivingProfile,
    DrivingProfileProcessor,
    LearnedBins,
)

from jepa_drive_wm.paths import OUTPUTS_DIR


SEQUENCE_NUMBERS = list(range(21))
MOTION_INTERVAL_S = 0.5
SEMANTIC_SAMPLE_S = 5.0
STATIONARY_MAX_MPS = 0.5

DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "kitti_profile_plots"


def build_profiles() -> tuple[LearnedBins, list[DrivingProfile]]:
    """Build exactly the same dataset bins and driving profiles as the analyser."""
    binner = DatasetBinner(
        SEQUENCE_NUMBERS,
        motion_interval_s=MOTION_INTERVAL_S,
        semantic_sample_s=SEMANTIC_SAMPLE_S,
        stationary_max_mps=STATIONARY_MAX_MPS,
    )
    bins = binner.fit()

    processor = DrivingProfileProcessor(
        bins,
        binner.measurements,
        turn_gap_s=MOTION_INTERVAL_S,
    )
    return bins, processor.build_all()


def _sorted_profiles(profiles: list[DrivingProfile]) -> list[DrivingProfile]:
    return sorted(profiles, key=lambda p: p.sequence_nr)


def _sequence_labels(profiles: list[DrivingProfile]) -> list[str]:
    return [f"{p.sequence_nr:02d}" for p in profiles]


def _finish_figure(fig: plt.Figure, path: Path, *, show: bool) -> None:
    """Save one complete figure and optionally display it interactively."""
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_speed_regimes(
    profiles: list[DrivingProfile],
    path: Path,
    *,
    show: bool = False,
) -> None:
    """Stacked horizontal bars showing how each sequence spends its driving time."""
    profiles = _sorted_profiles(profiles)
    labels = _sequence_labels(profiles)
    y = np.arange(len(profiles))

    components = [
        ("Stationary", np.array([p.stationary_fraction for p in profiles]) * 100.0),
        ("Slow", np.array([p.slow_fraction for p in profiles]) * 100.0),
        ("Medium", np.array([p.medium_fraction for p in profiles]) * 100.0),
        ("Fast", np.array([p.fast_fraction for p in profiles]) * 100.0),
    ]

    fig, ax = plt.subplots(figsize=(10, 9))
    left = np.zeros(len(profiles))

    for name, values in components:
        ax.barh(y, values, left=left, label=name)
        left += values

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xlabel("Fraction of motion windows")
    ax.set_ylabel("KITTI sequence")
    ax.set_title("Driving-speed composition by KITTI sequence")
    ax.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    _finish_figure(fig, path, show=show)


def plot_turn_counts(
    profiles: list[DrivingProfile],
    path: Path,
    *,
    show: bool = False,
) -> None:
    """Simple count plot for discrete strong-turn episodes."""
    profiles = _sorted_profiles(profiles)
    labels = _sequence_labels(profiles)
    y = np.arange(len(profiles))
    turns = np.array([p.turn_count for p in profiles])

    fig, ax = plt.subplots(figsize=(9, 9))
    bars = ax.barh(y, turns)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Detected turns")
    ax.set_ylabel("KITTI sequence")
    ax.set_title("Route complexity: number of strong turns")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)

    # Counts are the point of this plot, so print them directly at each bar end.
    for bar, count in zip(bars, turns):
        ax.text(
            bar.get_width() + 0.25,
            bar.get_y() + bar.get_height() / 2,
            str(int(count)),
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    _finish_figure(fig, path, show=show)


def plot_yaw_variability(
    profiles: list[DrivingProfile],
    path: Path,
    *,
    show: bool = False,
) -> None:
    """Plot gentle/background route curvature separately from discrete turns."""
    profiles = _sorted_profiles(profiles)
    labels = _sequence_labels(profiles)
    y = np.arange(len(profiles))
    yaw_std_deg_s = np.degrees(
        np.array([p.background_yaw_std_rad_s for p in profiles])
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(y, yaw_std_deg_s)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Background yaw variability (deg/s)")
    ax.set_ylabel("KITTI sequence")
    ax.set_title("Gentle road curvature after strong turns are removed")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    _finish_figure(fig, path, show=show)


def plot_environment_map(
    profiles: list[DrivingProfile],
    bins: LearnedBins,
    path: Path,
    *,
    show: bool = False,
) -> None:
    """Building-versus-vegetation map showing the four environment quadrants."""
    profiles = _sorted_profiles(profiles)

    building = np.array([p.building_fraction for p in profiles]) * 100.0
    vegetation = np.array([p.vegetation_fraction for p in profiles]) * 100.0
    building_threshold = bins.building_threshold * 100.0
    vegetation_threshold = bins.vegetation_threshold * 100.0

    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate scatter calls let Matplotlib's normal colour cycle distinguish the
    # four human-readable classes without hard-coding a visual theme.
    environment_classes = ("urban", "rural", "open-road", "mixed")
    for environment in environment_classes:
        selected = [p for p in profiles if p.environment_class == environment]
        if not selected:
            continue
        ax.scatter(
            [p.building_fraction * 100.0 for p in selected],
            [p.vegetation_fraction * 100.0 for p in selected],
            s=70,
            label=environment,
        )

    # Sequence numbers make every point identifiable without looking up a table.
    for p in profiles:
        ax.annotate(
            f"{p.sequence_nr:02d}",
            (p.building_fraction * 100.0, p.vegetation_fraction * 100.0),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axvline(building_threshold, linestyle="--", linewidth=1.2)
    ax.axhline(vegetation_threshold, linestyle="--", linewidth=1.2)

    x_max = max(55.0, float(building.max()) * 1.12)
    y_max = max(60.0, float(vegetation.max()) * 1.12)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Label the actual decision regions. These are descriptive annotations only;
    # the classification itself remains in DrivingProfileProcessor.
    left_x = building_threshold / 2.0
    right_x = building_threshold + (x_max - building_threshold) / 2.0
    lower_y = vegetation_threshold / 2.0
    upper_y = vegetation_threshold + (y_max - vegetation_threshold) / 2.0

    ax.text(left_x, lower_y, "OPEN-ROAD", ha="center", va="center", alpha=0.55)
    ax.text(left_x, upper_y, "RURAL", ha="center", va="center", alpha=0.55)
    ax.text(right_x, lower_y, "URBAN", ha="center", va="center", alpha=0.55)
    ax.text(right_x, upper_y, "MIXED", ha="center", va="center", alpha=0.55)

    ax.set_xlabel("Mean building pixels")
    ax.set_ylabel("Mean vegetation pixels")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_title("KITTI environment profiles from OneFormer semantics")
    ax.legend(title="Environment")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    _finish_figure(fig, path, show=show)


def make_all_plots(
    profiles: list[DrivingProfile],
    bins: LearnedBins,
    output_dir: Path,
    *,
    show: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_speed_regimes(profiles, output_dir / "speed_regimes.png", show=show)
    plot_turn_counts(profiles, output_dir / "turn_counts.png", show=show)
    plot_yaw_variability(profiles, output_dir / "yaw_variability.png", show=show)
    plot_environment_map(profiles, bins, output_dir / "environments.png", show=show)


def main() -> None:
    bins, profiles = build_profiles()
    make_all_plots(profiles, bins, DEFAULT_OUTPUT_DIR, show=True)


if __name__ == "__main__":
    main()
