"""Simple deterministic RGB forward projection with PyTorch3D.

Convention
----------
The source camera at time t is the world frame, using KITTI/OpenCV camera axes:

    +x = right, +y = down, +z = forward

A proposed future ego motion is

    [forward, right, yaw_right]

in metres / radians, relative to the source camera. Positive yaw_right is a
right turn: positive rotation about +y, rotating +z towards +x.

The module is dataset-agnostic. It knows only K, RGB, depth, and proposed ego
motion. PyTorch3D handles unprojection, camera projection and point rasterizing.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.renderer.cameras import get_screen_to_ndc_transform
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import cameras_from_opencv_projection


@dataclass
class WarpOutput:
    """Output of one proposed future warp."""

    rgb: torch.Tensor             # (H, W, 3): raw splatted RGB
    rgb_patch_masked: torch.Tensor  # invalid whole patches set to zero
    pixel_valid: torch.Tensor     # (H, W): raw geometric occupancy
    depth: torch.Tensor           # (H, W): target-camera z of winning point
    patch_coverage: torch.Tensor  # (H/P, W/P): fraction of occupied pixels
    patch_valid: torch.Tensor     # (H/P, W/P): coverage >= threshold


class PointRasterizer(nn.Module):
    """Hard nearest-depth point splatter using PyTorch3D PointsRasterizer."""

    def __init__(self, image_size: tuple[int, int], radius_px: float = 0.75):
        super().__init__()
        self.H, self.W = image_size

        # PyTorch3D point radius is in NDC. For rectangular images the shorter
        # image dimension spans 2 NDC units.
        radius_ndc = 2.0 * radius_px / min(self.H, self.W)

        settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=radius_ndc,
            points_per_pixel=1,  # one nearest point = hard z-buffer
            bin_size=None,
        )
        self.rasterizer = PointsRasterizer(
            cameras=None,
            raster_settings=settings,
        )

    def forward(
        self,
        point_cloud: Pointclouds,
        target_camera,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return warped_rgb, pixel_valid, warped_depth."""
        fragments = self.rasterizer(
            point_cloud,
            cameras=target_camera,
            eps=1e-8,
        )

        # points_per_pixel=1 -> take the only/nearest point at each pixel.
        idx = fragments.idx[0, ..., 0].long()       # (H, W)
        pixel_valid = idx >= 0

        colours = point_cloud.features_packed()     # (N_points, 3)
        warped_rgb = colours.new_zeros((self.H, self.W, 3))
        warped_rgb[pixel_valid] = colours[idx[pixel_valid]]

        # PointsRasterizer keeps target-view z in zbuf.
        warped_depth = fragments.zbuf[0, ..., 0]
        warped_depth = torch.where(
            pixel_valid,
            warped_depth,
            torch.zeros_like(warped_depth),
        )

        return warped_rgb, pixel_valid, warped_depth


class WarpModule(nn.Module):
    """Forward-project one RGB-D observation under proposed ego motion.

    Inputs
    ------
    rgb         : (H, W, 3), float
    depth       : (H, W), metric z-depth
    ego_motion  : (3,) = [forward, right, yaw_right]

    The source camera is treated as the world origin. A proposed ego motion is
    converted to the corresponding future camera pose; PyTorch3D then renders
    the static coloured point cloud from that future camera.
    """

    def __init__(
        self,
        intrinsics: torch.Tensor,
        image_size: tuple[int, int],
        *,
        radius_px: float = 0.75,
        patch_size: int = 16,
        patch_coverage_threshold: float = 0.60,
        min_depth: float = 1e-3,
    ):
        super().__init__()

        K = torch.as_tensor(intrinsics, dtype=torch.float32)
        if K.shape != (3, 3):
            raise ValueError(f"Expected K shape (3,3), got {tuple(K.shape)}")

        H, W = image_size
        if H % patch_size or W % patch_size:
            raise ValueError("Image dimensions must be divisible by patch_size.")

        self.register_buffer("K", K)
        self.H, self.W = image_size
        self.patch_size = patch_size
        self.patch_coverage_threshold = patch_coverage_threshold
        self.min_depth = min_depth

        self.rasterizer = PointRasterizer(image_size, radius_px)

    # ---------------------------------------------------------------------
    # Ego motion -> future camera
    # ---------------------------------------------------------------------

    @staticmethod
    def ego_motion_to_camera_pose(ego_motion: torch.Tensor) -> torch.Tensor:
        """Create T_source_target: pose of future camera in source coordinates.

        p_source = T_source_target @ p_target

        If ego = [f, r, psi], the future camera centre is [r, 0, f] in the
        source frame and its orientation is R_y(psi).
        """
        if ego_motion.shape != (3,):
            raise ValueError("ego_motion must be [forward, right, yaw_right].")

        forward, right, yaw = ego_motion.unbind()
        c, s = torch.cos(yaw), torch.sin(yaw)
        zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)

        # +yaw about +y rotates +z towards +x -> right turn.
        R_source_target = torch.stack([
            torch.stack([c,    zero, s]),
            torch.stack([zero, one,  zero]),
            torch.stack([-s,   zero, c]),
        ])
        C_source = torch.stack([right, zero, forward])

        T_source_target = torch.eye(
            4, device=ego_motion.device, dtype=ego_motion.dtype
        )
        T_source_target[:3, :3] = R_source_target
        T_source_target[:3, 3] = C_source
        return T_source_target

    @staticmethod
    def _invert_se3(T: torch.Tensor) -> torch.Tensor:
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = torch.eye(4, device=T.device, dtype=T.dtype)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -(R.T @ t)
        return T_inv

    @classmethod
    def ego_motion_to_warp_se3(cls, ego_motion: torch.Tensor) -> torch.Tensor:
        """Create T_target_source used to express static points in target view."""
        return cls._invert_se3(cls.ego_motion_to_camera_pose(ego_motion))

    # ---------------------------------------------------------------------
    # PyTorch3D camera / point-cloud construction
    # ---------------------------------------------------------------------

    def _camera_from_opencv_extrinsics(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
    ):
        """Build an equivalent PyTorch3D camera for X_cam = R @ X_world + t."""
        image_size = self.K.new_tensor([[self.H, self.W]])
        return cameras_from_opencv_projection(
            R=R[None],
            tvec=t[None],
            camera_matrix=self.K[None],
            image_size=image_size,
        )

    def _source_camera(self):
        R = torch.eye(3, device=self.K.device, dtype=self.K.dtype)
        t = torch.zeros(3, device=self.K.device, dtype=self.K.dtype)
        return self._camera_from_opencv_extrinsics(R, t)

    def _target_camera(self, ego_motion: torch.Tensor):
        T_target_source = self.ego_motion_to_warp_se3(ego_motion)
        return self._camera_from_opencv_extrinsics(
            T_target_source[:3, :3],
            T_target_source[:3, 3],
        )

    def _rgbd_to_pointcloud(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> Pointclouds:
        """Unproject source RGB-D to a coloured point cloud in source/world coords."""
        rgb = rgb.to(device=self.K.device, dtype=self.K.dtype)
        depth = depth.to(device=self.K.device, dtype=self.K.dtype)

        valid = (
            torch.isfinite(depth)
            & (depth > self.min_depth)
            & torch.isfinite(rgb).all(dim=-1)
        )
        if not valid.any():
            raise ValueError("Depth map contains no valid points.")

        # Ordinary image coordinates: u right, v down.
        v, u = torch.meshgrid(
            torch.arange(self.H, device=self.K.device, dtype=self.K.dtype),
            torch.arange(self.W, device=self.K.device, dtype=self.K.dtype),
            indexing="ij",
        )
        screen_xyz = torch.stack(
            [u[valid], v[valid], depth[valid]], dim=-1
        )[None]  # (1, N, 3)

        camera = self._source_camera()

        # cameras_from_opencv_projection returns an NDC PyTorch3D camera.
        # Convert image (+x right,+y down) -> PyTorch3D NDC (+x left,+y up).
        screen_to_ndc = get_screen_to_ndc_transform(
            camera,
            with_xyflip=True,
            image_size=((self.H, self.W),),
        )
        ndc_xyz = screen_to_ndc.transform_points(screen_xyz)

        # PyTorch3D inverts the perspective camera using metric view-space z.
        xyz_world = camera.unproject_points(
            ndc_xyz,
            world_coordinates=True,
            from_ndc=True,
        )[0]

        return Pointclouds(
            points=[xyz_world],
            features=[rgb[valid]],
        )

    # ---------------------------------------------------------------------
    # Geometric patch validity
    # ---------------------------------------------------------------------

    def _make_patch_mask(
        self,
        pixel_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute patch coverage from the untouched rasterizer occupancy mask."""
        P = self.patch_size

        coverage = F.avg_pool2d(
            pixel_valid.float()[None, None],
            kernel_size=P,
            stride=P,
        )[0, 0]
        patch_valid = coverage >= self.patch_coverage_threshold

        # Expand back to pixel resolution so whole invalid patches can be zeroed.
        pixel_patch_valid = patch_valid.repeat_interleave(P, 0).repeat_interleave(P, 1)
        return coverage, patch_valid, pixel_patch_valid

    def _render_pointcloud(
        self,
        point_cloud: Pointclouds,
        ego_motion: torch.Tensor,
    ) -> WarpOutput:
        target_camera = self._target_camera(ego_motion)
        rgb, pixel_valid, depth = self.rasterizer(point_cloud, target_camera)

        coverage, patch_valid, pixel_patch_valid = self._make_patch_mask(pixel_valid)

        # V0: reject low-coverage patches completely; do no RGB hole filling yet.
        rgb_patch_masked = rgb.clone()
        rgb_patch_masked[~pixel_patch_valid] = 0.0

        return WarpOutput(
            rgb=rgb,
            rgb_patch_masked=rgb_patch_masked,
            pixel_valid=pixel_valid,
            depth=depth,
            patch_coverage=coverage,
            patch_valid=patch_valid,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        ego_motion: torch.Tensor,
    ) -> WarpOutput:
        """Warp one source frame under one proposed future ego motion."""
        self._check_image_shapes(rgb, depth)
        ego_motion = torch.as_tensor(
            ego_motion, device=self.K.device, dtype=self.K.dtype
        )
        point_cloud = self._rgbd_to_pointcloud(rgb, depth)
        return self._render_pointcloud(point_cloud, ego_motion)

    def warp_sequence(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        ego_motions: Sequence[torch.Tensor] | torch.Tensor,
    ) -> list[WarpOutput]:
        """Warp one source frame to several proposed future views.

        ego_motions are all relative to the same source frame t:
            [ego(t->t+1), ego(t->t+2), ..., ego(t->t+K)]
        not incremental motions between adjacent future frames.
        """
        self._check_image_shapes(rgb, depth)
        point_cloud = self._rgbd_to_pointcloud(rgb, depth)  # build once

        outputs = []
        for ego_motion in ego_motions:
            ego_motion = torch.as_tensor(
                ego_motion, device=self.K.device, dtype=self.K.dtype
            )
            outputs.append(self._render_pointcloud(point_cloud, ego_motion))
        return outputs

    def _check_image_shapes(self, rgb: torch.Tensor, depth: torch.Tensor) -> None:
        if rgb.shape != (self.H, self.W, 3):
            raise ValueError(
                f"Expected rgb {(self.H, self.W, 3)}, got {tuple(rgb.shape)}"
            )
        if depth.shape != (self.H, self.W):
            raise ValueError(
                f"Expected depth {(self.H, self.W)}, got {tuple(depth.shape)}"
            )


# Example Usage
if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    from jepa_drive_wm.data.kitti import KITTISequence

    # -------------------------------------------------------------
    # Build a tube of N_OBSERVED context frames + N_FUTURE futures,
    # spaced FRAME_STRIDE apart, e.g. N_OBSERVED=4, N_FUTURE=4:
    #
    # Observed:  0, 5, 10, 15
    # Future:   20, 25, 30, 35
    #
    # We reconstruct the scene once from the last observed frame,
    # then render that point cloud from each proposed future pose.
    # -------------------------------------------------------------

    N_OBSERVED = 2
    N_FUTURE = 4
    FRAME_STRIDE = 5

    sequence = KITTISequence(sequence_nr=16)
    frame_ids = [200 + FRAME_STRIDE * i for i in range(N_OBSERVED + N_FUTURE)]
    observed_ids = frame_ids[:N_OBSERVED]
    future_ids = frame_ids[N_OBSERVED:]
    source_id = observed_ids[-1]

    n_cols = max(N_OBSERVED, N_FUTURE)

    # -------------------------------------------------------------
    # 1. Plot the ground-truth RGB tube (observed row, future row).
    # -------------------------------------------------------------

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, frame_id in zip(axes[0], observed_ids):
        ax.imshow(sequence.get_image(frame_id))
        ax.set_title(f"GT observed {frame_id}")
    for ax, frame_id in zip(axes[1], future_ids):
        ax.imshow(sequence.get_image(frame_id))
        ax.set_title(f"GT future {frame_id}")

    fig.suptitle("Ground-truth RGB tube", fontsize=14)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # 2. Warp the final observed frame into each future frame using
    #    only the proposed ego motion [forward, right, yaw_right].
    # -------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_rgb = torch.from_numpy(sequence.get_image(source_id)).float().to(device)
    source_depth = torch.from_numpy(sequence.get_depth(source_id)).float().to(device)

    # Resize to 384x1248 so both dimensions are divisible by patch_size=16,
    # scaling the intrinsics to match.
    orig_H, orig_W = source_rgb.shape[:2]
    H, W = 384, 1248
    source_rgb = F.interpolate(
        source_rgb.permute(2, 0, 1)[None],
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    source_depth = F.interpolate(
        source_depth[None, None],
        size=(H, W),
        mode="nearest",
    )[0, 0]

    K = sequence.calib.K2.clone()
    K[0] *= W / orig_W
    K[1] *= H / orig_H

    warper = WarpModule(
        intrinsics=K,
        image_size=(H, W),
        radius_px=1.0,
        patch_size=16,
        patch_coverage_threshold=0.7,
    ).to(device)

    # Each motion is source-relative: source_id -> future_id.
    ego_motions = [
        torch.from_numpy(sequence.get_ego_motion(source_id, future_id)).float().to(device)
        for future_id in future_ids
    ]

    # Time the warp alone: sync before/after so pending GPU work is counted.
    if device.type == "cuda":
        torch.cuda.synchronize()
    warp_start = time.perf_counter()

    with torch.no_grad():
        warped_outputs = warper.warp_sequence(
            rgb=source_rgb,
            depth=source_depth,
            ego_motions=ego_motions,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    warp_seconds = time.perf_counter() - warp_start
    print(
        f"Warp time ({device.type}): {warp_seconds * 1e3:.1f} ms total "
        f"for {N_FUTURE} futures "
        f"({warp_seconds * 1e3 / N_FUTURE:.1f} ms per future frame)"
    )

    # -------------------------------------------------------------
    # 3. Plot observed RGB followed by projected future RGB.
    #    rgb_patch_masked blacks out whole 16x16 patches with <80%
    #    genuine rasterizer coverage.
    # -------------------------------------------------------------

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    # Top row: real observed frames.
    for ax, frame_id in zip(axes[0], observed_ids):
        ax.imshow(sequence.get_image(frame_id))
        ax.set_title(f"Observed frame {frame_id}")

    # Bottom row: deterministic forward projections from source_id.
    for ax, future_id, ego_motion, output in zip(
        axes[1], future_ids, ego_motions, warped_outputs
    ):
        warped_rgb = output.rgb_patch_masked.detach().cpu().clamp(0, 1).numpy()
        forward, right, yaw_right = ego_motion.detach().cpu().tolist()

        ax.imshow(warped_rgb)
        ax.set_title(
            f"Warped → {future_id}\n"
            f"f={forward:.2f} m, r={right:.2f} m, yaw={yaw_right:.3f} rad"
        )
        ax.axis("off")

    fig.suptitle(
        f"Observed frames + forward projections from frame {source_id}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------
    # Optional: direct GT-vs-warp comparison for each future frame.
    # -------------------------------------------------------------

    fig, axes = plt.subplots(2, N_FUTURE, figsize=(4.5 * N_FUTURE, 6), squeeze=False)

    for col, (future_id, output) in enumerate(zip(future_ids, warped_outputs)):
        axes[0, col].imshow(sequence.get_image(future_id))
        axes[0, col].set_title(f"GT frame {future_id}")
        axes[0, col].axis("off")

        warped_rgb = output.rgb_patch_masked.detach().cpu().clamp(0, 1).numpy()
        axes[1, col].imshow(warped_rgb)
        axes[1, col].set_title(f"Warped from {source_id}")
        axes[1, col].axis("off")

    fig.suptitle("Future ground truth vs deterministic forward projection", fontsize=14)
    plt.tight_layout()
    plt.show()