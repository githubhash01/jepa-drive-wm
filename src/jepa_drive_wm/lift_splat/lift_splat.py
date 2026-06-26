"""
Lifts and Splats FV to BEV


Given FV-RGB or FV-VJEPA features:

- Lift: project each valid pixel/feature into 3D space using the depth map and camera intrinsics

        (u, v, RGB) -> (x, y, z, RGB) or (u, v, VJEPA) -> (x, y, z, VJEPA)

- Splat: project the 3D points into the BEV plane and accumulate features into a BEV grid

        (x, y, z, RGB) -> (X, Z, RGB) or (x, y, z, VJEPA) -> (X, Z, VJEPA)

Design note:
    There is not really an "RGB lifter" and a "V-JEPA lifter". There is ONE geometric
    lifter (this base class) and the subclasses only decide what value rides along with
    each lifted 3D point. RGB attaches a pixel colour; V-JEPA attaches the feature of
    the patch that contains the pixel. The geometry is identical.

KITTI camera convention used throughout: x = right, y = down, z = forward.
BEV grid uses (x, z); y is only used for height filtering.
"""

from typing import Callable
import numpy as np
import torch


class LiftSplat:
    """Shared geometry: back-project -> filter -> BEV cells -> scatter-mean splat."""

    def __init__(
        self,
        K_image: torch.Tensor,                          # intrinsics e.g. K2 or K3 (3, 3) in image coords
        x_range: tuple[float, float] = (-30.0, 30.0),   # lateral, metres
        z_range: tuple[float, float] = (0.0, 50.0),     # forward, metres
        y_range: tuple[float, float] = (-3.0, 1.0),     # height band kept (y is DOWN)
        resolution: float = 0.05,                        # metres per BEV cell
        depth_scale: float = 1.0,                       # use 256.0 only when passing raw uint16 PNG values
    ):
        self.K_image = torch.as_tensor(K_image, dtype=torch.float32)
        self.x_range = x_range
        self.z_range = z_range
        self.y_range = y_range
        self.resolution = resolution
        self.depth_scale = depth_scale

        # Compute BEV grid dimensions
        self.bev_width = int((x_range[1] - x_range[0]) / resolution)
        self.bev_height = int((z_range[1] - z_range[0]) / resolution)

    # --- geometry -----------------------------------------------------------

    def _backproject(self, depth: torch.Tensor):
        """
        depth: (H, W) metric depth by default. Divided by depth_scale if raw values are passed.
        Returns 3D camera points for every valid pixel, plus their (u, v) indices.
            points: (M, 3) as (x, y, z)
            us, vs: (M,) long pixel columns/rows (kept so subclasses can gather values)
        """
        depth = torch.as_tensor(depth, dtype=torch.float32)
        H, W = depth.shape
        z = depth / self.depth_scale

        fx, fy = self.K_image[0, 0], self.K_image[1, 1]
        cx, cy = self.K_image[0, 2], self.K_image[1, 2]

        vs, us = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        valid = z > 0
        us, vs, z = us[valid], vs[valid], z[valid]

        x = (us.float() - cx) * z / fx
        y = (vs.float() - cy) * z / fy
        points = torch.stack([x, y, z], dim=1)
        return points, us, vs

    def _filter(self, points: torch.Tensor) -> torch.Tensor:
        """Boolean mask keeping points inside x/y/z ranges."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        return (
            (x >= self.x_range[0]) & (x < self.x_range[1])
            & (y >= self.y_range[0]) & (y < self.y_range[1])
            & (z >= self.z_range[0]) & (z < self.z_range[1])
        )

    def _to_cells(self, points: torch.Tensor) -> torch.Tensor:
        """Metric (x, z) -> flat BEV cell index (row * bev_width + col)."""
        x, z = points[:, 0], points[:, 2]
        col = ((x - self.x_range[0]) / self.resolution).floor().long()
        row = ((self.z_range[1] - z) / self.resolution).floor().long()  # far->top, near->bottom
        col = col.clamp(0, self.bev_width - 1)
        row = row.clamp(0, self.bev_height - 1)
        return row * self.bev_width + col

    def _splat(self, cells: torch.Tensor, values: torch.Tensor):
        """
        Scatter-mean values into the BEV grid.
            values: (N, C)
        Returns:
            bev:    (bev_height, bev_width, C) mean value per cell (0 where unobserved)
            counts: (bev_height, bev_width)
        """
        C = values.shape[1]
        n_cells = self.bev_height * self.bev_width

        sums = torch.zeros(n_cells, C, dtype=torch.float32)
        counts = torch.zeros(n_cells, dtype=torch.float32)
        sums.index_add_(0, cells, values.float())
        counts.index_add_(0, cells, torch.ones_like(cells, dtype=torch.float32))

        bev = sums / counts.clamp(min=1).unsqueeze(1)
        return bev.reshape(self.bev_height, self.bev_width, C), counts.reshape(self.bev_height, self.bev_width)

    def _splat_top(self, cells: torch.Tensor, values: torch.Tensor, points: torch.Tensor):
        """
        RGB-friendly splat: for each BEV cell, keep the highest point.
        KITTI camera convention has y down, so "highest" means smallest y.

        counts still counts all points that landed in the cell.
        """
        C = values.shape[1]
        n_cells = self.bev_height * self.bev_width

        counts = torch.zeros(n_cells, dtype=torch.float32, device=values.device)
        counts.index_add_(0, cells, torch.ones_like(cells, dtype=torch.float32))

        # Sort by cell, then by -y so that within each cell the smallest y comes last.
        # BIG only needs to be larger than the possible y spread.
        y = points[:, 1]
        BIG = 1000.0
        sort_key = cells.double() * BIG - y.double()
        order = torch.argsort(sort_key)

        cells_sorted = cells[order]
        values_sorted = values[order]

        # Last item in each cell group is the highest point.
        is_last = torch.ones_like(cells_sorted, dtype=torch.bool)
        is_last[:-1] = cells_sorted[:-1] != cells_sorted[1:]

        selected_cells = cells_sorted[is_last]
        selected_values = values_sorted[is_last]

        bev = torch.zeros(n_cells, C, dtype=torch.float32, device=values.device)
        bev[selected_cells] = selected_values.float()

        return (
            bev.reshape(self.bev_height, self.bev_width, C),
            counts.reshape(self.bev_height, self.bev_width),
        )

    def _lift_splat(
        self,
        depth: torch.Tensor,
        gather_values: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        splat_mode: str = "mean",
    ):
        points, us, vs = self._backproject(depth)
        keep = self._filter(points)
        points, us, vs = points[keep], us[keep], vs[keep]

        values = gather_values(us, vs)
        cells = self._to_cells(points)

        if splat_mode == "mean":
            bev, counts = self._splat(cells, values)
        elif splat_mode == "top":
            bev, counts = self._splat_top(cells, values, points)
        else:
            raise ValueError(f"Unknown splat_mode: {splat_mode}")

        mask = counts > 0
        return bev, counts, mask

class LiftSplatRGB(LiftSplat):
    """My values are RGB pixels."""

    def lift_splat(self, image, depth, splat_mode: str = "top"):
        image = torch.as_tensor(np.asarray(image))
        image = image.float() / 255.0 if image.dtype == torch.uint8 else image.float()
        return self._lift_splat(
            depth,
            lambda us, vs: image[vs, us],
            splat_mode=splat_mode,
        )


class LiftSplatVJEPA(LiftSplat):
    """
    Two distinct V-JEPA lifting strategies — call the one you want explicitly.

        lift_splat_pixels_with_patch_tokens  (dense)
            Every valid depth pixel is lifted into 3D. Each pixel inherits the
            feature of whichever V-JEPA patch contains it. Good for filling the BEV.
            Odd note: the same patch token gets copied across all its pixels.

        lift_splat_patch_tokens  (sparse, conceptually clean)
            Each V-JEPA patch becomes exactly one 3D point. The patch-centre ray
            is back-projected at the median depth within that patch. No feature is
            ever duplicated; each token appears at most once in the BEV.
    """

    def lift_splat_pixels_with_patch_tokens(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
        grid_hw: tuple[int, int] = (24, 78),
    ):
        """
        Dense V-JEPA splat: many pixels → many 3D points, each carrying its patch's feature.

            features: (num_patches, D) V-JEPA patch tokens.
            depth:    (H0, W0) metric depth, same resolution as the original image.
            grid_hw:  (grid_h, grid_w) V-JEPA patch grid.
        Returns (bev_features (H_bev, W_bev, D), counts, observed_mask).
        """
        features = torch.as_tensor(features)
        depth = torch.as_tensor(depth, dtype=torch.float32)
        H0, W0 = depth.shape
        grid_h, grid_w = grid_hw

        def gather(us, vs):
            # Map pixel (u,v) -> patch index. patch_size cancels: idx = floor(px * grid / image).
            pu = (us.float() * grid_w / W0).floor().long().clamp(0, grid_w - 1)
            pv = (vs.float() * grid_h / H0).floor().long().clamp(0, grid_h - 1)
            return features[pv * grid_w + pu]

        return self._lift_splat(depth, gather)

    def lift_splat_patch_tokens(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
        grid_hw: tuple[int, int] = (24, 78),
    ):
        """
        Sparse V-JEPA splat: one 3D point per patch → each token appears at most once in BEV.
        The patch-centre ray is back-projected at the median depth within the patch footprint.
        Patches with no valid depth are skipped entirely.

            features: (num_patches, D) V-JEPA patch tokens.
            depth:    (H0, W0) metric depth, same resolution as the original image.
            grid_hw:  (grid_h, grid_w) V-JEPA patch grid.
        Returns (bev_features (H_bev, W_bev, D), counts, observed_mask).
        """
        features = torch.as_tensor(features, dtype=torch.float32)
        depth = torch.as_tensor(depth, dtype=torch.float32)
        H0, W0 = depth.shape
        grid_h, grid_w = grid_hw

        fx, fy = self.K_image[0, 0], self.K_image[1, 1]
        cx, cy = self.K_image[0, 2], self.K_image[1, 2]

        patch_h = H0 / grid_h  # patch footprint in original-image pixels (float ok)
        patch_w = W0 / grid_w

        points_list, feature_list = [], []

        for pv in range(grid_h):
            for pu in range(grid_w):
                # Pixel bounds of this patch in the original image.
                r0 = int(pv * patch_h);       r1 = min(int((pv + 1) * patch_h), H0)
                c0 = int(pu * patch_w);       c1 = min(int((pu + 1) * patch_w), W0)

                patch_depth = depth[r0:r1, c0:c1]
                valid = patch_depth > 0
                if not valid.any():
                    continue

                # Median depth of valid pixels inside this patch.
                z = float(patch_depth[valid].median())

                # Back-project the patch-centre ray.
                u_c = (c0 + c1) / 2.0
                v_c = (r0 + r1) / 2.0
                x = (u_c - float(cx)) * z / float(fx)
                y = (v_c - float(cy)) * z / float(fy)

                points_list.append(torch.tensor([x, y, z]))
                feature_list.append(features[pv * grid_w + pu])

        if not points_list:
            D = features.shape[1]
            empty = torch.zeros(self.bev_height, self.bev_width, D)
            empty_counts = torch.zeros(self.bev_height, self.bev_width)
            return empty, empty_counts, empty_counts > 0

        points = torch.stack(points_list)          # (num_valid_patches, 3)
        values = torch.stack(feature_list)         # (num_valid_patches, D)

        keep = self._filter(points)
        if not keep.any():
            D = features.shape[1]
            empty = torch.zeros(self.bev_height, self.bev_width, D)
            empty_counts = torch.zeros(self.bev_height, self.bev_width)
            return empty, empty_counts, empty_counts > 0

        cells = self._to_cells(points[keep])
        bev, counts = self._splat(cells, values[keep])
        mask = counts > 0
        return bev, counts, mask
    
