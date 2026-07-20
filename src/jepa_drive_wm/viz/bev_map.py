"""
Accumulate lifted, coloured KITTI observations into a world-frame BEV map.

This is the SLAM-style "map building" backend for the sequence visualiser. For
each frame it:

    1. back-projects the depth map to 3D points in the left camera (cam2),
       keeping only a height band + a near-depth cut (far stereo depth is noisy);
    2. colours each point -- either by its left-image RGB, or by its coarse
       planning group (taxonomy) -- ;
    3. transforms the points cam2 -> world using the sequence calibration + GT
       pose (world = cam0 of frame 0, KITTI odometry convention);
    4. scatter-adds the colours/one-hots into one persistent world grid.

The ground is the world x-z plane, so the map is indexed (world z -> row,
world x -> col). Read the growing map with `world_image()` (global map) or
`ego_image(i)` (crop re-centred on the car at frame i).

Only the lift + height/depth filter are borrowed from `LiftSplat`; the lateral/
forward extent is handled by the world grid, not the per-frame camera grid.
"""

import os

import cv2
import numpy as np
import torch

from jepa_drive_wm.lift_splat.lift_splat import LiftSplat
from jepa_drive_wm.probes.dense.taxonomy import (
    CLASS_TO_GROUP, GROUP_NAMES, GROUP_PALETTE, NUM_GROUPS,
)

# Points in this coarse group are moving agents (car/person/... -> "dynamic
# object"). A static world accumulation smears them into trails, so we never
# bake them into the persistent map; they are drawn as a current-frame overlay.
DYNAMIC_GROUP = GROUP_NAMES.index("dynamic object")

_G = {name: i for i, name in enumerate(GROUP_NAMES)}
# For a planning-style BEV, static structure + sky are visual clutter. Show only
# the drivable surface, its soft margins, and moving agents; everything else
# falls through to the background. Override `mapper.coarse_visible` to change.
DEFAULT_COARSE_VISIBLE = (_G["drivable"], _G["soft-drivable"], _G["dynamic object"])


class BEVMapper:
    """World-frame BEV accumulator for one KITTI sequence.

    Usage: `reset(mode)` picks the value mode ("rgb" or "coarse"), then call
    `add_frame(i)` for each frame and read `world_image()` / `ego_image(i)` as
    the map grows.
    """

    _CHANNELS = {"rgb": 3, "coarse": NUM_GROUPS}

    def __init__(self, sequence, *, resolution: float = 0.2, max_depth: float = 30.0,
                 y_range: tuple[float, float] = (-2.5, 1.0), margin: float = 20.0,
                 mask_dynamic: bool = True) -> None:
        self.seq = sequence
        self.resolution = float(resolution)
        # Keep moving agents out of the persistent map (drawn live instead).
        self.mask_dynamic = mask_dynamic
        # Coarse groups actually drawn (others -> background). Planning-relevant only.
        self.coarse_visible = np.asarray(DEFAULT_COARSE_VISIBLE)

        # LiftSplat is used only for camera-frame back-projection + the height/
        # depth band filter. x is left wide open (world extent handled below);
        # depth_scale=1.0 because seq.get_depth already returns metric metres.
        self.geom = LiftSplat(
            sequence.calib.K2.float(),
            x_range=(-1e9, 1e9),
            z_range=(0.5, max_depth),
            y_range=y_range,
            resolution=resolution,
            depth_scale=1.0,
        )

        # World grid sized to the trajectory (cam0 positions) plus a margin.
        t = sequence.gt_poses[:, :3, 3]
        self.x_min = float(t[:, 0].min()) - margin
        self.x_max = float(t[:, 0].max()) + margin
        self.z_min = float(t[:, 2].min()) - margin
        self.z_max = float(t[:, 2].max()) + margin
        self.map_w = int(round((self.x_max - self.x_min) / self.resolution))
        self.map_h = int(round((self.z_max - self.z_min) / self.resolution))

        self.mode = "rgb"
        self.reset("rgb")

    # ------------------------------------------------------------------ #
    # accumulation
    # ------------------------------------------------------------------ #

    def reset(self, mode: str = "rgb") -> None:
        """Clear the map and set the value mode ("rgb" or "coarse")."""
        if mode not in self._CHANNELS:
            raise ValueError(f"mode must be one of {list(self._CHANNELS)}, got {mode!r}")
        self.mode = mode
        n = self.map_h * self.map_w
        self.sums = torch.zeros(n, self._CHANNELS[mode], dtype=torch.float32)
        self.counts = torch.zeros(n, dtype=torch.float32)
        self._clear_dynamic()

    def frame_available(self, i: int) -> bool:
        """True if the inputs needed by the current mode exist for frame i."""
        if not os.path.exists(self.seq.depths[i]):
            return False
        if self.mode == "coarse":
            from jepa_drive_wm.data.kitti import SEMANTICS_DIR
            sem_path = SEMANTICS_DIR / f"{self.seq.sequence_nr:02d}" / f"{i:06d}.png"
            return sem_path.exists()
        return True

    def _load_semantics(self, i: int) -> np.ndarray | None:
        """Coarse group map (H, W) for frame i, or None if the PNG is missing."""
        from jepa_drive_wm.data.kitti import SEMANTICS_DIR
        p = SEMANTICS_DIR / f"{self.seq.sequence_nr:02d}" / f"{i:06d}.png"
        if not p.exists():
            return None
        return self.seq.get_semantics(i)

    def _values(self, i: int, sem: np.ndarray | None,
                us: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
        """Per-point value vectors for the current mode at pixels (us, vs)."""
        if self.mode == "rgb":
            img = torch.from_numpy(self.seq.get_image(i)).float()   # (H, W, 3) in [0, 1]
            return img[vs, us]                                      # (M, 3)
        # coarse: one-hot over planning groups so a per-cell argmax = majority vote.
        groups = CLASS_TO_GROUP[sem][vs.numpy(), us.numpy()]        # (M,)
        onehot = np.zeros((groups.shape[0], NUM_GROUPS), dtype=np.float32)
        onehot[np.arange(groups.shape[0]), groups] = 1.0
        return torch.from_numpy(onehot)

    # --- transient (current-frame-only) dynamic-agent overlay ----------------

    def _clear_dynamic(self) -> None:
        self._dyn_cells = None
        self._dyn_colors = None

    def _set_dynamic(self, cells: torch.Tensor, values: torch.Tensor) -> None:
        """Store this frame's dynamic-agent cells + display colours (overwrites)."""
        if self.mode == "rgb":
            colors = (values.clamp(0, 1).numpy() * 255).astype(np.uint8)      # true colour
        else:
            colors = np.broadcast_to(GROUP_PALETTE[DYNAMIC_GROUP],
                                     (cells.shape[0], 3)).copy()              # taxonomy red
        self._dyn_cells = cells
        self._dyn_colors = colors

    def _world_cells(self, points_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World points (M, 3) -> (kept idx, flat cell index) inside the grid."""
        x, z = points_w[:, 0], points_w[:, 2]
        keep = ((x >= self.x_min) & (x < self.x_max)
                & (z >= self.z_min) & (z < self.z_max))
        idx = keep.nonzero(as_tuple=True)[0]
        col = ((x[idx] - self.x_min) / self.resolution).long().clamp_(0, self.map_w - 1)
        row = ((self.z_max - z[idx]) / self.resolution).long().clamp_(0, self.map_h - 1)
        return idx, row * self.map_w + col

    def add_frame(self, i: int) -> bool:
        """Lift frame i, colour the points, splat into the world map. False if skipped.

        Static points reinforce the persistent map; dynamic-agent points (when
        `mask_dynamic`) are held out and exposed as a current-frame overlay so
        moving cars track their live position instead of smearing a trail.
        """
        if not os.path.exists(self.seq.depths[i]):
            self._clear_dynamic()
            return False

        depth = torch.from_numpy(self.seq.get_depth(i))
        points, us, vs = self.geom._backproject(depth)
        keep = self.geom._filter(points)
        points, us, vs = points[keep], us[keep], vs[keep]
        if points.shape[0] == 0:
            self._clear_dynamic()
            return False

        # Semantics drive coarse colouring and/or dynamic masking.
        sem = self._load_semantics(i) if (self.mode == "coarse" or self.mask_dynamic) else None
        if self.mode == "coarse" and sem is None:
            self._clear_dynamic()
            return False   # coarse can't colour without semantics

        values = self._values(i, sem, us, vs)

        # cam2 -> world via the sequence calibration + GT pose (both float64).
        ones = torch.ones(points.shape[0], 1, dtype=torch.float64)
        p_c2 = torch.cat([points.double(), ones], dim=1)            # (M, 4) homogeneous
        points_w = self.seq.cam2_point_to_world(p_c2, i)[:, :3]     # (M, 3) world

        idx, cells = self._world_cells(points_w)
        values = values[idx]

        dyn = None
        if sem is not None and self.mask_dynamic:
            groups = CLASS_TO_GROUP[sem][vs.numpy(), us.numpy()]    # (M,) over filtered pts
            dyn = torch.from_numpy(groups[idx.numpy()] == DYNAMIC_GROUP)

        if dyn is not None and bool(dyn.any()):
            stat = ~dyn
            self.sums.index_add_(0, cells[stat], values[stat])
            self.counts.index_add_(0, cells[stat], torch.ones(int(stat.sum())))
            self._set_dynamic(cells[dyn], values[dyn])
        else:
            self.sums.index_add_(0, cells, values)
            self.counts.index_add_(0, cells, torch.ones(cells.shape[0]))
            self._clear_dynamic()
        return True

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #

    def world_image(self, background: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Current accumulated map as an (H, W, 3) uint8 RGB image."""
        occ = self.counts > 0
        img = np.empty((self.map_h * self.map_w, 3), np.uint8)
        img[:] = np.asarray(background, np.uint8)
        if occ.any():
            occ_np = occ.numpy()
            if self.mode == "rgb":
                mean = (self.sums[occ] / self.counts[occ].unsqueeze(1)).clamp(0, 1)
                img[occ_np] = (mean.numpy() * 255).astype(np.uint8)
            else:  # coarse: majority group per cell, but only the visible groups
                win = self.sums[occ].argmax(1).numpy()
                vis = np.isin(win, self.coarse_visible)
                rows = np.nonzero(occ_np)[0][vis]
                img[rows] = GROUP_PALETTE[win[vis]]

        # Live dynamic-agent overlay (current frame only) painted on top.
        if self._dyn_cells is not None and self._dyn_cells.numel():
            img[self._dyn_cells.numpy()] = self._dyn_colors
        return img.reshape(self.map_h, self.map_w, 3)

    def world_px(self, x: float, z: float) -> tuple[float, float]:
        """World metric (x, z) -> world-image pixel (col, row)."""
        return ((x - self.x_min) / self.resolution,
                (self.z_max - z) / self.resolution)

    def pose_pixels(self, upto: int | None = None) -> tuple[list[int], list[int]]:
        """Trajectory poses [0:upto] as world-image (rows, cols) integer pixel lists."""
        t = self.seq.gt_poses[:upto, :3, 3]
        cols = ((t[:, 0] - self.x_min) / self.resolution).long().tolist()
        rows = ((self.z_max - t[:, 2]) / self.resolution).long().tolist()
        return rows, cols

    def ego_image(self, i: int, *, fwd: float = 30.0, back: float = 10.0,
                  side: float = 20.0, background: tuple[int, int, int] = (0, 0, 0),
                  draw_ego: bool = True) -> np.ndarray:
        """
        The accumulated map re-centred on the car at frame i: ego at bottom-centre,
        forward pointing up. Window spans [-side, side] laterally and [-back, fwd]
        along the heading. Produced by an affine warp of the world map, so the car
        stays fixed while the map slides/rotates under it.
        """
        world = self.world_image(background=background)
        res = self.resolution

        pose = self.seq.gt_poses[i]
        ex, ez = float(pose[0, 3]), float(pose[2, 3])       # ego position in world x-z
        fx, fz = float(pose[0, 2]), float(pose[2, 2])       # cam0 forward (z) in world x-z

        ew = int(round((2 * side) / res))
        eh = int(round((fwd + back) / res))

        def ego_px(x: float, z: float) -> tuple[float, float]:
            dx, dz = x - ex, z - ez
            xe = dx * fz - dz * fx      # lateral (right), forward-relative
            ze = dx * fx + dz * fz      # along heading
            return ((xe + side) / res, (fwd - ze) / res)

        # 3 world anchors -> their world-px (src) and ego-px (dst) => affine warp.
        anchors = [(ex, ez), (ex + 10.0, ez), (ex, ez + 10.0)]
        src = np.float32([self.world_px(x, z) for x, z in anchors])
        dst = np.float32([ego_px(x, z) for x, z in anchors])
        M = cv2.getAffineTransform(src, dst)
        ego = cv2.warpAffine(world, M, (ew, eh), flags=cv2.INTER_NEAREST,
                             borderValue=background)

        if draw_ego:
            cx, cy = int(side / res), int(fwd / res)        # ego at (0, 0) in ego metres
            s = max(3, int(round(1.2 / res)))
            tri = np.array([[cx, cy - s], [cx - s, cy + s], [cx + s, cy + s]], np.int32)
            cv2.fillConvexPoly(ego, tri, (255, 255, 255))   # up-pointing = forward
        return ego
