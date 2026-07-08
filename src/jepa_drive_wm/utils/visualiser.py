"""
class Visualiser:

responsible for all visualisation of data and results

"""

import numpy as np
import matplotlib.pyplot as plt

# TODO - write global PCA myself
from jepa_drive_wm.data.global_pca import (
    load_pca_npz,
    transform_features_with_saved_pca,
    normalize_pca_rgb,
)

class Visualiser:

    def __init__(self, pca_path: str = "/home/hashim/Desktop/Outputs/vjepa21_global_train_pca.npz") -> None: # TODO - get rid of hardcoded path
        self.pca_path = pca_path
        self._pca_data = None

    def _pca(self):
        """Lazily load and cache the saved global PCA basis."""
        if self._pca_data is None:
            self._pca_data = load_pca_npz(self.pca_path)
        return self._pca_data


class FVVisualizer(Visualiser):
    """
    Visualiser for Front-View (FV) data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def display_rgb_image(self, rgb_image: np.ndarray | str, title: str = "RGB Image"):
        """
        Visualise an RGB image.
        """
        plt.figure(figsize=(10, 10))
        if isinstance(rgb_image, str):
            rgb_image = plt.imread(rgb_image)
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def display_depth_map(self, depth_map: np.ndarray | str, title: str = "Depth Map"):
        """
        Visualise a depth map.
        """
        plt.figure(figsize=(10, 10))
        if isinstance(depth_map, str):
            depth_map = plt.imread(depth_map)
        plt.imshow(depth_map, cmap='plasma')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def vjepa_embedding_pca(self, features, grid_hw = (24, 78)):
        """
        Project flat V-JEPA patch features (num_patches, D) into the global PCA RGB
        space and return them. grid_hw is (grid_h, grid_w) used to reshape into an image.
        """
        grid_h, grid_w = grid_hw
        pca = self._pca()
        Y = transform_features_with_saved_pca(features, pca)
        pca_rgb = normalize_pca_rgb(Y.reshape(grid_h, grid_w, 3), pca["rgb_low"], pca["rgb_high"])
        return pca_rgb
    
    def display_vjepa_embedding_pca(self, features, grid_hw = (24, 78), title: str = "VJEPA Embedding PCA"): # TODO - get rid of hardcoded grid_hw
        """
        Project flat V-JEPA patch features (num_patches, D) into the global PCA RGB
        space and display them. grid_hw is (grid_h, grid_w) used to reshape into an image.
        """
        # grid_h, grid_w = grid_hw
        # pca = self._pca()
        # Y = transform_features_with_saved_pca(features, pca)
        # pca_rgb = normalize_pca_rgb(Y.reshape(grid_h, grid_w, 3), pca["rgb_low"], pca["rgb_high"])
        pca_rgb = self.vjepa_embedding_pca(features, grid_hw)
        plt.figure(figsize=(10, 10))
        plt.imshow(pca_rgb, interpolation="nearest")
        plt.title(title)
        plt.axis("off")
        plt.show()


class BEVVisualizer(Visualiser):
    """
    Visualiser for Bird's Eye View (BEV) data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def display_bev_RGB_image(self, bev_image, title: str = "BEV RGB Image"):
        """
        Visualise a BEV RGB image. Accepts a numpy array or torch tensor (H, W, 3) in [0, 1].
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(np.asarray(bev_image), interpolation="nearest")
        plt.title(title)
        plt.axis('off')
        plt.show()

    def display_bev_vjepa_pca(self, bev_features, counts, title: str = "BEV VJEPA PCA"):
        """
        Project a BEV V-JEPA feature grid through the global PCA and show it as an RGB
        BEV image. Same idea as the FV PCA display, but the features live on the BEV grid
        instead of the patch grid. Unobserved cells (counts == 0) are left black.

            bev_features: (H_bev, W_bev, D)
            counts:       (H_bev, W_bev)
        """
        bev_features = np.asarray(bev_features, dtype=np.float32)
        observed = np.asarray(counts) > 0
        H, W, D = bev_features.shape

        pca = self._pca()
        # Same transform as transform_features_with_saved_pca, inlined for numpy input.
        Y = (bev_features.reshape(-1, D) - pca["mean"]) @ pca["components"].T
        pca_rgb = normalize_pca_rgb(Y.reshape(H, W, 3), pca["rgb_low"], pca["rgb_high"])
        pca_rgb[~observed] = 0.0  # unobserved != free space; keep it visibly empty

        plt.figure(figsize=(10, 10))
        plt.imshow(pca_rgb, interpolation="nearest")
        plt.title(title)
        plt.axis("off")
        plt.show()

    # ------------------------------------------------------------------
    # CARLA-style surround-fused BEV
    # ------------------------------------------------------------------
    def _fit_bev_pca_rgb(self, bev_features, observed, low_pct=1.0, high_pct=99.0):
        """Fit a fresh 3-component PCA on this BEV's OBSERVED cells and map to RGB.

        The saved global basis was fit on KITTI patches, so its colour percentiles
        do not transfer to CARLA features (washed-out output). For a self-contained
        CARLA view we fit PCA on the observed feature cells here and normalise with
        their own 1/99 percentiles -- vivid, but colours are only comparable within
        this frame. Use pca_mode="saved" if you want a fixed cross-frame basis.
        """
        H, W, D = bev_features.shape
        flat = bev_features.reshape(-1, D)
        X = flat[observed.reshape(-1)]                       # (N, D) observed only
        if X.shape[0] < 3:
            return np.zeros((H, W, 3), dtype=np.float32)

        mean = X.mean(axis=0)
        # Right singular vectors of the centred observed cells are the PCA axes.
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
        components = Vt[:3]                                  # (3, D)

        Y = (flat - mean) @ components.T                     # (H*W, 3)
        Y_obs = Y[observed.reshape(-1)]
        lo = np.percentile(Y_obs, low_pct, axis=0).astype(np.float32)
        hi = np.percentile(Y_obs, high_pct, axis=0).astype(np.float32)
        hi = np.where(hi > lo, hi, lo + 1e-6)
        return normalize_pca_rgb(Y.reshape(H, W, 3), lo, hi)

    def carla_bev_vjepa_pca(self, bev_features, counts, pca_mode: str = "fit"):
        """Return an (H, W, 3) RGB image of a CARLA surround-fused BEV feature grid.

        bev_features: (H, W, D)  from LiftSplatCarla.splat_vjepa_surround
        counts:       (H, W)
        pca_mode:     "fit"   -> per-frame PCA on observed cells (default, CARLA)
                      "saved" -> fixed global basis loaded from disk (KITTI-fit)

        Unobserved cells (counts == 0) are returned black.
        """
        bev_features = np.asarray(bev_features, dtype=np.float32)
        observed = np.asarray(counts) > 0
        H, W, D = bev_features.shape

        if pca_mode == "saved":
            pca = self._pca()
            Y = (bev_features.reshape(-1, D) - pca["mean"]) @ pca["components"].T
            pca_rgb = normalize_pca_rgb(Y.reshape(H, W, 3), pca["rgb_low"], pca["rgb_high"])
        elif pca_mode == "fit":
            pca_rgb = self._fit_bev_pca_rgb(bev_features, observed)
        else:
            raise ValueError(f"pca_mode must be 'fit' or 'saved', got {pca_mode!r}")

        pca_rgb = np.array(pca_rgb, dtype=np.float32)
        pca_rgb[~observed] = 0.0
        return pca_rgb

    def display_carla_bev_vjepa_pca(
        self,
        bev_features,
        counts,
        *,
        pca_mode: str = "saved",
        ego_extent: tuple[float, float] | None = None,
        x_range: tuple[float, float] = (-50.0, 50.0),
        y_range: tuple[float, float] = (-50.0, 50.0),
        resolution: float = 0.5,
        title: str = "CARLA Fused BEV V-JEPA PCA",
        ax=None,
    ):
        """Display a CARLA surround-fused BEV V-JEPA PCA, ego-centred and forward-up.

        Styled to match Sim/generate_bev.py: origin='lower' so ego forward points UP,
        ego drawn as a red box with a yellow heading arrow at the grid centre, and
        axis ticks in metres. The grid defaults match the label / lift-splat grid
        (x,y in [-50, 50] m at 0.5 m/cell -> 200 x 200), so this overlays cell-for-cell
        on the BEV labels.

            bev_features: (H, W, D)   features[r, c] with forward=+row, right=+col
            counts:       (H, W)
            ego_extent:   (ext_x, ext_y) ego half-sizes (m) from calibration.json
                          ego_bbox.extent_xyz; if None the ego marker is skipped.
        """
        from matplotlib.patches import Rectangle

        pca_rgb = self.carla_bev_vjepa_pca(bev_features, counts, pca_mode=pca_mode)
        H, W = pca_rgb.shape[:2]

        own_fig = ax is None
        if own_fig:
            _, ax = plt.subplots(figsize=(10, 10))

        # extent maps cell indices to metres; origin='lower' puts forward (+x) up.
        extent = (y_range[0], y_range[1], x_range[0], x_range[1])  # (left,right,bottom,top)
        ax.imshow(pca_rgb, origin="lower", interpolation="nearest", extent=extent)

        if ego_extent is not None:
            ext_x, ext_y = ego_extent           # forward half-length, lateral half-width
            ax.add_patch(Rectangle(
                (-ext_y, -ext_x), 2 * ext_y, 2 * ext_x,   # (y, x) lower-left in metres
                facecolor="red", edgecolor="white", lw=1.0, zorder=5,
            ))
            ax.arrow(0.0, 0.0, 0.0, max(ext_x * 1.6, 3.0), length_includes_head=True,
                     head_width=ext_y * 1.4, head_length=ext_x, fc="yellow", ec="yellow", zorder=6)

        ax.set_title(title)
        ax.set_xlabel("y right (m)")
        ax.set_ylabel("x forward (m)")
        if own_fig:
            plt.show()
        return ax