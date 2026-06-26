"""
BEV fusion utilities for pose-aware RGB / V-JEPA lift-splat observations.

Core idea
---------
A single lift-splat output is a BEV *observation* in the source camera frame.
Fusion turns a set of observations into a BEV *state* by warping observations into
one reference frame and doing weighted averaging with masks/confidence.

Two common reference-frame policies use the same machinery:

1. SLAM/map style:
      reference frame = world / map frame
      observations are fused into one persistent map.

2. Ego-centred state style:
      reference frame = current camera/ego frame
      recent observations are fused into a local state around the current frame.

This module assumes the BEV grid uses camera-like axes:
    x = right, y = down, z = forward
and BEV cells are laid out over (x, z). Height y is collapsed; warping therefore
uses a representative y value, usually y = 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import torch


WeightMode = Literal["counts", "binary", "sqrt_counts", "log_counts"]


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Inverse of a rigid 4x4 transform T^a_b -> T^b_a."""
    T = torch.as_tensor(T)
    R = T[:3, :3]
    t = T[:3, 3:4]
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R.transpose(-1, -2)
    T_inv[:3, 3:4] = -R.transpose(-1, -2) @ t
    return T_inv


@dataclass(frozen=True)
class BEVGridSpec:
    """Metric definition of a BEV grid.

    The grid indexes cells by (row, col), but the underlying metric coordinates
    are (x, z):
        col increases with x     left -> right
        row increases as z lowers far -> near, image-style display
    """

    x_range: Tuple[float, float] = (-20.0, 20.0)
    z_range: Tuple[float, float] = (0.0, 60.0)
    resolution: float = 0.05
    y_ref: float = 0.0

    @property
    def width(self) -> int:
        return int((self.x_range[1] - self.x_range[0]) / self.resolution)

    @property
    def height(self) -> int:
        return int((self.z_range[1] - self.z_range[0]) / self.resolution)

    @property
    def num_cells(self) -> int:
        return self.height * self.width

    def assert_hw(self, H: int, W: int) -> None:
        if H != self.height or W != self.width:
            raise ValueError(
                f"Grid shape mismatch: expected ({self.height}, {self.width}), got ({H}, {W})"
            )

    def cell_centres(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        *,
        y: Optional[float] = None,
        homogeneous: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """Return metric cell centres in this BEV frame.

        rows, cols: integer tensors with matching shape.
        Returns: (N, 4) homogeneous or (N, 3) non-homogeneous points.
        """
        rows = rows.to(device=device).long()
        cols = cols.to(device=device).long()
        y_val = self.y_ref if y is None else y

        x = self.x_range[0] + (cols.to(dtype) + 0.5) * self.resolution
        z = self.z_range[1] - (rows.to(dtype) + 0.5) * self.resolution
        yy = torch.full_like(x, float(y_val))
        pts = torch.stack([x, yy, z], dim=-1)
        if not homogeneous:
            return pts
        ones = torch.ones_like(x)[..., None]
        return torch.cat([pts, ones], dim=-1)

    def metric_to_cells(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert metric points in this frame to BEV row/col indices.

        points may be (..., 3) or (..., 4). Only x and z are used.
        Returns rows, cols, valid mask with leading shape (...).
        """
        x = points[..., 0]
        z = points[..., 2]

        valid = (
            (x >= self.x_range[0])
            & (x < self.x_range[1])
            & (z >= self.z_range[0])
            & (z < self.z_range[1])
        )

        cols = torch.floor((x - self.x_range[0]) / self.resolution).long()
        rows = torch.floor((self.z_range[1] - z) / self.resolution).long()
        cols = cols.clamp(0, self.width - 1)
        rows = rows.clamp(0, self.height - 1)
        return rows, cols, valid

    def flatten(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        return rows.long() * self.width + cols.long()


@dataclass
class BEVObservation:
    """Single-frame lift-splat output in one source frame, usually c2_i.

    features:      (H_bev, W_bev, C), mean value per observed cell
    counts:        (H_bev, W_bev), number/confidence of splatted points per cell
    grid:          metric BEV grid used to create this observation
    T_world_frame: 4x4 transform mapping this observation frame into world/map frame
                   e.g. T^w_c2_i for KITTI image_2 frame i.
    """

    features: torch.Tensor
    counts: torch.Tensor
    grid: BEVGridSpec
    T_world_frame: torch.Tensor
    name: str = ""

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError(f"features must be (H,W,C), got {tuple(self.features.shape)}")
        if self.counts.ndim != 2:
            raise ValueError(f"counts must be (H,W), got {tuple(self.counts.shape)}")
        H, W, _ = self.features.shape
        self.grid.assert_hw(H, W)
        if tuple(self.counts.shape) != (H, W):
            raise ValueError("features and counts spatial dimensions must match")
        self.T_world_frame = torch.as_tensor(
            self.T_world_frame, dtype=torch.float32, device=self.features.device
        )

    @property
    def observed(self) -> torch.Tensor:
        return self.counts > 0

    @property
    def channels(self) -> int:
        return int(self.features.shape[-1])


@dataclass
class BEVState:
    """Fused BEV state/map in a chosen reference frame.

    features:      (H_bev, W_bev, C), fused feature mean
    weights:       (H_bev, W_bev), accumulated confidence/weight
    age:           (H_bev, W_bev), fusion steps since last observation; -1 = never seen
    grid:          metric grid of the state reference frame
    T_world_frame: 4x4 transform mapping the state reference frame into world
                   map style: usually identity
                   ego style: T^w_c2_current
    """

    features: torch.Tensor
    weights: torch.Tensor
    age: torch.Tensor
    grid: BEVGridSpec
    T_world_frame: torch.Tensor
    name: str = ""

    @classmethod
    def empty(
        cls,
        *,
        grid: BEVGridSpec,
        channels: int,
        T_world_frame: torch.Tensor,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        name: str = "",
    ) -> "BEVState":
        H, W = grid.height, grid.width
        return cls(
            features=torch.zeros(H, W, channels, dtype=dtype, device=device),
            weights=torch.zeros(H, W, dtype=dtype, device=device),
            age=torch.full((H, W), -1, dtype=torch.long, device=device),
            grid=grid,
            T_world_frame=torch.as_tensor(T_world_frame, dtype=torch.float32, device=device),
            name=name,
        )

    @property
    def observed(self) -> torch.Tensor:
        return self.weights > 0

    @property
    def channels(self) -> int:
        return int(self.features.shape[-1])


@dataclass
class FusionConfig:
    """Simple deterministic fusion settings."""

    weight_mode: WeightMode = "counts"
    max_observation_weight: Optional[float] = 32.0
    state_decay: float = 1.0
    y_ref: float = 0.0


class BEVFuser:
    """Pose-aware fusion of BEV observations into BEV states."""

    def __init__(self, config: FusionConfig = FusionConfig()) -> None:
        self.config = config

    def _observation_weights(self, counts: torch.Tensor) -> torch.Tensor:
        counts = counts.float()
        if self.config.weight_mode == "counts":
            w = counts
        elif self.config.weight_mode == "binary":
            w = (counts > 0).float()
        elif self.config.weight_mode == "sqrt_counts":
            w = torch.sqrt(counts.clamp(min=0))
        elif self.config.weight_mode == "log_counts":
            w = torch.log1p(counts.clamp(min=0))
        else:
            raise ValueError(f"Unknown weight mode: {self.config.weight_mode}")

        if self.config.max_observation_weight is not None:
            w = w.clamp(max=float(self.config.max_observation_weight))
        return w

    def warp_observation_to_state_frame(
        self,
        obs: BEVObservation,
        state_grid: BEVGridSpec,
        T_state_obs: torch.Tensor,
        *,
        channels: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Warp one BEV observation into a target/state BEV frame.

        T_state_obs maps metric points from the observation frame into the state frame.

        Returns:
            warped_features: (H_state, W_state, C)
            warped_weights:  (H_state, W_state)
        """
        device = torch.device(device) if device is not None else obs.features.device
        dtype = dtype if dtype is not None else obs.features.dtype
        C = obs.channels if channels is None else channels

        if obs.channels != C:
            raise ValueError(f"Observation has {obs.channels} channels but state expects {C}")

        features = obs.features.to(device=device, dtype=dtype)
        counts = obs.counts.to(device=device)
        observed = counts > 0

        rows, cols = torch.nonzero(observed, as_tuple=True)
        n = int(rows.numel())
        Hs, Ws = state_grid.height, state_grid.width
        if n == 0:
            return (
                torch.zeros(Hs, Ws, C, dtype=dtype, device=device),
                torch.zeros(Hs, Ws, dtype=dtype, device=device),
            )

        values = features[rows, cols]  # (N, C)
        weights = self._observation_weights(counts[rows, cols]).to(device=device, dtype=dtype)

        pts_obs = obs.grid.cell_centres(
            rows,
            cols,
            y=self.config.y_ref,
            homogeneous=True,
            dtype=torch.float32,
            device=device,
        )  # (N, 4)

        T_state_obs = torch.as_tensor(T_state_obs, dtype=torch.float32, device=device)
        pts_state = (T_state_obs @ pts_obs.T).T
        tgt_rows, tgt_cols, valid = state_grid.metric_to_cells(pts_state)

        if not valid.any():
            return (
                torch.zeros(Hs, Ws, C, dtype=dtype, device=device),
                torch.zeros(Hs, Ws, dtype=dtype, device=device),
            )

        values = values[valid]
        weights = weights[valid]
        flat = state_grid.flatten(tgt_rows[valid], tgt_cols[valid])

        sums = torch.zeros(state_grid.num_cells, C, dtype=dtype, device=device)
        weight_sums = torch.zeros(state_grid.num_cells, dtype=dtype, device=device)
        sums.index_add_(0, flat, values * weights.unsqueeze(-1))
        weight_sums.index_add_(0, flat, weights)

        warped = sums / weight_sums.clamp(min=1e-6).unsqueeze(-1)
        warped[weight_sums == 0] = 0
        return warped.reshape(Hs, Ws, C), weight_sums.reshape(Hs, Ws)

    def fuse(self, state: BEVState, obs: BEVObservation) -> BEVState:
        """Fuse one observation into a state.

        The observation and state may live in different reference frames. Their
        T_world_frame matrices define the required warp:

            T_state_obs = inv(T_world_state) @ T_world_obs
        """
        device = state.features.device
        dtype = state.features.dtype

        if obs.channels != state.channels:
            raise ValueError(f"Channel mismatch: obs={obs.channels}, state={state.channels}")

        T_world_state = state.T_world_frame.to(device=device, dtype=torch.float32)
        T_world_obs = obs.T_world_frame.to(device=device, dtype=torch.float32)
        T_state_obs = se3_inverse(T_world_state) @ T_world_obs

        obs_features, obs_weights = self.warp_observation_to_state_frame(
            obs,
            state.grid,
            T_state_obs,
            channels=state.channels,
            device=device,
            dtype=dtype,
        )

        old_weights = state.weights.to(device=device, dtype=dtype) * float(self.config.state_decay)
        old_features = state.features.to(device=device, dtype=dtype)
        denom = old_weights + obs_weights

        new_features = torch.zeros_like(old_features)
        valid = denom > 0
        new_features[valid] = (
            old_features[valid] * old_weights[valid].unsqueeze(-1)
            + obs_features[valid] * obs_weights[valid].unsqueeze(-1)
        ) / denom[valid].unsqueeze(-1)

        # Age convention: -1 = never seen; 0 = observed on this fusion step.
        new_age = state.age.to(device=device).clone()
        seen_before = old_weights > 0
        new_age[seen_before] = torch.clamp(new_age[seen_before], min=0) + 1
        new_age[obs_weights > 0] = 0
        new_age[denom == 0] = -1

        return BEVState(
            features=new_features,
            weights=denom,
            age=new_age,
            grid=state.grid,
            T_world_frame=T_world_state,
            name=state.name,
        )


class EgoCentredBEVFusion:
    """Fuse a short window of observations into the current observation's frame."""

    def __init__(self, fuser: Optional[BEVFuser] = None) -> None:
        self.fuser = fuser if fuser is not None else BEVFuser()

    def fuse_window(
        self,
        observations: Sequence[BEVObservation],
        *,
        target_index: int = -1,
        grid: Optional[BEVGridSpec] = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> BEVState:
        if len(observations) == 0:
            raise ValueError("Need at least one observation")

        target = observations[target_index]
        grid = grid if grid is not None else target.grid
        state = BEVState.empty(
            grid=grid,
            channels=target.channels,
            T_world_frame=target.T_world_frame,
            device=device,
            dtype=dtype,
            name=f"ego_state@{target.name}" if target.name else "ego_state",
        )

        for obs in observations:
            state = self.fuser.fuse(state, obs)
        return state


class GlobalBEVMapFusion:
    """Maintain a persistent fixed-size map in a chosen world/map frame."""

    def __init__(
        self,
        *,
        grid: BEVGridSpec,
        channels: int,
        T_world_map: Optional[torch.Tensor] = None,
        fuser: Optional[BEVFuser] = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        name: str = "global_map",
    ) -> None:
        self.fuser = fuser if fuser is not None else BEVFuser()
        T_world_map = torch.eye(4) if T_world_map is None else T_world_map
        self.state = BEVState.empty(
            grid=grid,
            channels=channels,
            T_world_frame=T_world_map,
            device=device,
            dtype=dtype,
            name=name,
        )

    def fuse(self, obs: BEVObservation) -> BEVState:
        self.state = self.fuser.fuse(self.state, obs)
        return self.state

    def fuse_many(self, observations: Sequence[BEVObservation]) -> BEVState:
        for obs in observations:
            self.fuse(obs)
        return self.state


# KITTI convenience ----------------------------------------------------------

def T_world_c2_from_kitti(seq, frame_idx: int) -> torch.Tensor:
    """Return T^w_c2_i for a KITTISequence-like object.

    Requires:
        seq.pose_cam0_to_world(i) -> T^w_c0_i
        seq.calib.T_c0_c2       -> T^c0_c2
    """
    return (seq.pose_cam0_to_world(frame_idx).float() @ seq.calib.T_c0_c2.float()).float()