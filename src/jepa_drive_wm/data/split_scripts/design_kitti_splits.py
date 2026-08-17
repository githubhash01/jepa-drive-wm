"""KITTI driving-profile analysis, version 2.

This file intentionally has a new name and a new output banner so it cannot be
confused with earlier split-design experiments.  It does not choose a train /
validation / test split yet; it only measures each drive and learns a small set
of dataset-informed profile bins.

Design used in this version:
  * Stationary is physical: speed <= 0.5 m/s.
  * Only moving samples are clustered into slow / medium / fast.
  * A turn is one contiguous episode in the strongest learned yaw-rate regime.
  * OneFormer is sampled every 5 s.
  * Environment uses only building and vegetation occupancy.
  * High building + high vegetation is reported as ``mixed`` rather than being
    forced into urban or rural.

The code is split into two working classes:
  DatasetBinner             learns shared dataset-wide boundaries.
  DrivingProfileProcessor   converts each measured sequence into a profile.

Dataclasses are passive records for measurements, learned boundaries, and final
profiles; they contain no hidden classification logic.

When this exact version runs, its first lines identify both the profile version
and the source path.  This is deliberate: if an older similarly named script is
still present on a workstation, the terminal output makes that immediately
visible before any learned statistics are printed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jepa_drive_wm.data.kitti import KITTISequence


# Cityscapes train IDs produced by the OneFormer preprocessing pipeline.
CITYSCAPES_BUILDING_ID = 2
CITYSCAPES_VEGETATION_ID = 8

PROFILE_VERSION = "2026-08-17-v2"
PROFILE_SIGNATURE = "physical-stationary / moving-k3 / mixed-environment"
SPEED_LABELS = ("stationary", "slow", "medium", "fast")


@dataclass(frozen=True)
class SequenceMeasurements:
    """Raw measurements used to construct one sequence profile."""

    sequence_nr: int
    duration_s: float
    motion_times_s: np.ndarray
    speed_mps: np.ndarray
    yaw_rate_rad_s: np.ndarray  # signed
    building_fraction: np.ndarray
    vegetation_fraction: np.ndarray


@dataclass(frozen=True)
class LearnedBins:
    """Dataset-wide boundaries learned before any sequence is classified."""

    stationary_max_mps: float
    moving_speed_centres_mps: tuple[float, float, float]
    moving_speed_edges_mps: tuple[float, float]

    yaw_rate_centres_rad_s: tuple[float, float, float]
    strong_turn_threshold_rad_s: float

    building_centres: tuple[float, float]
    building_threshold: float

    vegetation_centres: tuple[float, float]
    vegetation_threshold: float


@dataclass(frozen=True)
class DrivingProfile:
    """Compact, human-readable description of one KITTI drive."""

    sequence_nr: int
    duration_s: float

    stationary_fraction: float
    slow_fraction: float
    medium_fraction: float
    fast_fraction: float
    speed_class: str

    turn_count: int
    background_yaw_std_rad_s: float

    building_fraction: float
    vegetation_fraction: float
    environment_class: str


@dataclass(frozen=True)
class _KMeans1DResult:
    centres: np.ndarray
    edges: np.ndarray


def _fit_1d_kmeans(values: np.ndarray, k: int, max_iter: int = 100) -> _KMeans1DResult:
    """Small deterministic 1-D k-means implementation with no extra dependency."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Cannot fit bins to an empty array.")
    if np.unique(x).size < k:
        raise ValueError(f"Need at least {k} distinct values to fit {k} bins.")

    # Quantile initialisation is deterministic and spreads centres across the data.
    quantiles = (np.arange(k, dtype=np.float64) + 0.5) / k
    centres = np.quantile(x, quantiles)

    for _ in range(max_iter):
        labels = np.argmin(np.abs(x[:, None] - centres[None, :]), axis=1)
        updated = centres.copy()
        for cluster in range(k):
            members = x[labels == cluster]
            if members.size:
                updated[cluster] = members.mean()

        if np.allclose(updated, centres, rtol=0.0, atol=1e-10):
            centres = updated
            break
        centres = updated

    centres = np.sort(centres)
    edges = (centres[:-1] + centres[1:]) / 2.0
    return _KMeans1DResult(centres=centres, edges=edges)


class DatasetBinner:
    """Measure the dataset once and learn all dataset-dependent bin boundaries."""

    def __init__(
        self,
        sequence_numbers: list[int],
        *,
        motion_interval_s: float = 0.5,
        semantic_sample_s: float = 5.0,
        stationary_max_mps: float = 0.5,
    ) -> None:
        if motion_interval_s <= 0:
            raise ValueError("motion_interval_s must be positive.")
        if semantic_sample_s <= 0:
            raise ValueError("semantic_sample_s must be positive.")
        if stationary_max_mps <= 0:
            raise ValueError("stationary_max_mps must be positive.")

        self.sequence_numbers = list(sequence_numbers)
        self.motion_interval_s = float(motion_interval_s)
        self.semantic_sample_s = float(semantic_sample_s)
        self.stationary_max_mps = float(stationary_max_mps)
        self.measurements: dict[int, SequenceMeasurements] = {}

    def fit(self) -> LearnedBins:
        """Collect sequence measurements, then learn the shared bin boundaries."""
        self.measurements = {
            seq_nr: self._measure_sequence(KITTISequence(sequence_nr=seq_nr))
            for seq_nr in self.sequence_numbers
        }

        speed = np.concatenate([m.speed_mps for m in self.measurements.values()])
        abs_yaw_rate = np.concatenate(
            [np.abs(m.yaw_rate_rad_s) for m in self.measurements.values()]
        )

        # Environment bins are learned from one mean occupancy value per sequence,
        # so every drive contributes equally regardless of its duration.
        building_means = np.array(
            [m.building_fraction.mean() for m in self.measurements.values()]
        )
        vegetation_means = np.array(
            [m.vegetation_fraction.mean() for m in self.measurements.values()]
        )

        # "Stationary" should mean near-zero physical motion, not merely the
        # slowest cluster present in KITTI. Learn slow/medium/fast only from
        # samples that are actually moving.
        moving_speed = speed[speed > self.stationary_max_mps]
        speed_bins = _fit_1d_kmeans(moving_speed, k=3)
        yaw_bins = _fit_1d_kmeans(abs_yaw_rate, k=3)
        building_bins = _fit_1d_kmeans(building_means, k=2)
        vegetation_bins = _fit_1d_kmeans(vegetation_means, k=2)

        return LearnedBins(
            stationary_max_mps=self.stationary_max_mps,
            moving_speed_centres_mps=tuple(float(x) for x in speed_bins.centres),
            moving_speed_edges_mps=tuple(float(x) for x in speed_bins.edges),
            yaw_rate_centres_rad_s=tuple(float(x) for x in yaw_bins.centres),
            # The strongest yaw-rate cluster represents discrete turns; the
            # midpoint to the middle cluster is the learned entry threshold.
            strong_turn_threshold_rad_s=float(yaw_bins.edges[-1]),
            building_centres=tuple(float(x) for x in building_bins.centres),
            building_threshold=float(building_bins.edges[0]),
            vegetation_centres=tuple(float(x) for x in vegetation_bins.centres),
            vegetation_threshold=float(vegetation_bins.edges[0]),
        )

    def _measure_sequence(self, seq: KITTISequence) -> SequenceMeasurements:
        times, speed, yaw_rate = self._measure_motion(seq)
        building, vegetation = self._measure_environment(seq)

        return SequenceMeasurements(
            sequence_nr=seq.sequence_nr,
            duration_s=float(seq.duration),
            motion_times_s=times,
            speed_mps=speed,
            yaw_rate_rad_s=yaw_rate,
            building_fraction=building,
            vegetation_fraction=vegetation,
        )

    def _measure_motion(
        self, seq: KITTISequence
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sliding ~0.5 s motion estimates, starting from every source frame."""
        sample_times: list[float] = []
        speeds: list[float] = []
        yaw_rates: list[float] = []

        for i in range(seq.nr_frames - 1):
            target_time = seq.times[i] + self.motion_interval_s
            j = int(np.searchsorted(seq.times, target_time, side="left"))
            if j >= seq.nr_frames:
                break

            dt = float(seq.times[j] - seq.times[i])
            if dt <= 0:
                continue

            forward, right, yaw = seq.get_ego_motion(i, j)
            speeds.append(float(np.hypot(forward, right) / dt))
            yaw_rates.append(float(yaw / dt))
            sample_times.append(float((seq.times[i] + seq.times[j]) / 2.0))

        if not speeds:
            raise ValueError(f"Sequence {seq.sequence_nr:02d} has no valid motion windows.")

        return (
            np.asarray(sample_times, dtype=np.float64),
            np.asarray(speeds, dtype=np.float64),
            np.asarray(yaw_rates, dtype=np.float64),
        )

    def _measure_environment(self, seq: KITTISequence) -> tuple[np.ndarray, np.ndarray]:
        """Sample OneFormer maps every few seconds and retain only two classes."""
        target_times = np.arange(
            seq.times[0],
            seq.times[-1] + 1e-9,
            self.semantic_sample_s,
            dtype=np.float64,
        )
        frame_indices = np.searchsorted(seq.times, target_times, side="left")
        frame_indices = np.unique(np.clip(frame_indices, 0, seq.nr_frames - 1))

        building: list[float] = []
        vegetation: list[float] = []

        for frame_index in frame_indices:
            labels = seq.get_semantics(int(frame_index))
            building.append(float(np.mean(labels == CITYSCAPES_BUILDING_ID)))
            vegetation.append(float(np.mean(labels == CITYSCAPES_VEGETATION_ID)))

        if not building:
            raise ValueError(
                f"Sequence {seq.sequence_nr:02d} has no semantic samples."
            )

        return (
            np.asarray(building, dtype=np.float64),
            np.asarray(vegetation, dtype=np.float64),
        )


class DrivingProfileProcessor:
    """Convert raw measurements into compact profiles using learned bins."""

    def __init__(
        self,
        bins: LearnedBins,
        measurements: dict[int, SequenceMeasurements],
        *,
        turn_gap_s: float = 0.5,
    ) -> None:
        self.bins = bins
        self.measurements = measurements
        self.turn_gap_s = float(turn_gap_s)

    def build_all(self) -> list[DrivingProfile]:
        return [self.build(seq_nr) for seq_nr in sorted(self.measurements)]

    def build(self, sequence_nr: int) -> DrivingProfile:
        m = self.measurements[sequence_nr]

        # Bin 0 is true near-zero motion. Bins 1..3 are the learned
        # slow/medium/fast regimes among moving samples.
        speed_bin = np.zeros(m.speed_mps.shape, dtype=np.int64)
        moving = m.speed_mps > self.bins.stationary_max_mps
        speed_bin[moving] = 1 + np.digitize(
            m.speed_mps[moving], self.bins.moving_speed_edges_mps
        )
        fractions = np.bincount(speed_bin, minlength=4).astype(np.float64)
        fractions /= fractions.sum()
        dominant_speed = SPEED_LABELS[int(np.argmax(fractions))]

        strong_turn = np.abs(m.yaw_rate_rad_s) >= self.bins.strong_turn_threshold_rad_s
        turn_count = self._count_turn_episodes(m.motion_times_s, strong_turn)

        background_yaw = m.yaw_rate_rad_s[~strong_turn]
        background_yaw_std = (
            float(np.std(background_yaw)) if background_yaw.size else 0.0
        )

        building = float(m.building_fraction.mean())
        vegetation = float(m.vegetation_fraction.mean())
        environment = self._environment_class(building, vegetation)

        return DrivingProfile(
            sequence_nr=sequence_nr,
            duration_s=m.duration_s,
            stationary_fraction=float(fractions[0]),
            slow_fraction=float(fractions[1]),
            medium_fraction=float(fractions[2]),
            fast_fraction=float(fractions[3]),
            speed_class=dominant_speed,
            turn_count=turn_count,
            background_yaw_std_rad_s=background_yaw_std,
            building_fraction=building,
            vegetation_fraction=vegetation,
            environment_class=environment,
        )

    def _count_turn_episodes(self, times: np.ndarray, strong_turn: np.ndarray) -> int:
        """Count strong-yaw episodes, merging gaps below the motion resolution."""
        turn_times = times[strong_turn]
        if turn_times.size == 0:
            return 0

        # A new turn starts only when the strong-yaw samples have been separated
        # by more than the 0.5 s motion horizon. Shorter gaps are below the
        # temporal resolution of the yaw estimate and are treated as one turn.
        return 1 + int(np.count_nonzero(np.diff(turn_times) > self.turn_gap_s))

    def _environment_class(self, building: float, vegetation: float) -> str:
        """Classify the two learned semantic axes without forcing ambiguity.

        High building + low vegetation  -> urban
        Low building  + high vegetation -> rural
        Low building  + low vegetation  -> open-road
        High building + high vegetation -> mixed

        The mixed quadrant is useful for residential/suburban drives where the
        scene contains substantial built structure and substantial greenery.
        """
        built_up = building >= self.bins.building_threshold
        vegetated = vegetation >= self.bins.vegetation_threshold

        if built_up and vegetated:
            return "mixed"
        if built_up:
            return "urban"
        if vegetated:
            return "rural"
        return "open-road"


def print_bins(bins: LearnedBins) -> None:
    """Show the learned boundaries explicitly so the classification is inspectable."""
    print("Learned dataset bins")
    print("--------------------")
    print(f"stationary threshold (m/s): {bins.stationary_max_mps:.2f}")
    print(
        "moving speed centres (m/s): "
        + ", ".join(f"{x:.2f}" for x in bins.moving_speed_centres_mps)
    )
    print(
        "moving speed bounds (m/s):  "
        + ", ".join(f"{x:.2f}" for x in bins.moving_speed_edges_mps)
    )
    print(
        "|yaw rate| centres (rad/s): "
        + ", ".join(f"{x:.3f}" for x in bins.yaw_rate_centres_rad_s)
    )
    print(f"strong-turn threshold:      {bins.strong_turn_threshold_rad_s:.3f} rad/s")
    print(
        f"building centres:           {bins.building_centres[0]:.1%}, "
        f"{bins.building_centres[1]:.1%}"
    )
    print(f"urban building threshold:   {bins.building_threshold:.1%}")
    print(
        f"vegetation centres:         {bins.vegetation_centres[0]:.1%}, "
        f"{bins.vegetation_centres[1]:.1%}"
    )
    print(f"rural vegetation threshold: {bins.vegetation_threshold:.1%}")


def print_profiles(profiles: list[DrivingProfile]) -> None:
    header = (
        f"{'seq':>3} | {'stat':>5} {'slow':>5} {'med':>5} {'fast':>5} {'speed':>10} | "
        f"{'turns':>5} {'yaw_std':>7} | {'build':>6} {'veg':>6} {'environment':>11}"
    )
    print(header)
    print("-" * len(header))

    for p in profiles:
        # Degrees/s is easier to read in the report; the dataclass retains rad/s.
        yaw_std_deg_s = np.degrees(p.background_yaw_std_rad_s)
        print(
            f"{p.sequence_nr:>3} | "
            f"{p.stationary_fraction:5.0%} {p.slow_fraction:5.0%} "
            f"{p.medium_fraction:5.0%} {p.fast_fraction:5.0%} {p.speed_class:>10} | "
            f"{p.turn_count:5d} {yaw_std_deg_s:7.2f} | "
            f"{p.building_fraction:6.1%} {p.vegetation_fraction:6.1%} "
            f"{p.environment_class:>11}"
        )

    print("\nyaw_std is background signed-yaw variability in deg/s after strong turns are removed.")


def main() -> None:
    print(f"KITTI driving profiles [{PROFILE_VERSION}]")
    print(f"source file: {__file__}")
    print(f"profile signature: {PROFILE_SIGNATURE}")
    print()

    sequence_numbers = list(range(22))

    binner = DatasetBinner(
        sequence_numbers,
        motion_interval_s=0.5,
        semantic_sample_s=5.0,
        stationary_max_mps=0.5,
    )
    bins = binner.fit()

    processor = DrivingProfileProcessor(
        bins,
        binner.measurements,
        turn_gap_s=0.5,
    )
    profiles = processor.build_all()

    print_bins(bins)
    print()
    print_profiles(profiles)


if __name__ == "__main__":
    main()
