"""
class Visualiser:

responsible for all visualisation of data and results

"""

import numpy as np
import matplotlib.pyplot as plt
from jepa_drive_wm.probes.dense.taxonomy import PALETTE, CLASS_TO_GROUP, GROUP_PALETTE, NUM_CLASSES, NUM_GROUPS


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

    def display_semantic_map(self, semantic_map: np.ndarray | str, title: str = "Semantic Map"):
        """
        Visualise a semantic segmentation map using the Cityscapes palette.
        """
        plt.figure(figsize=(10, 10))
        if isinstance(semantic_map, str):
            semantic_map = plt.imread(semantic_map)
        # Map class ids to colors
        color_map = PALETTE[semantic_map]
        plt.imshow(color_map)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def display_semantic_course(self, semantic_map: np.ndarray | str, title: str = "Semantic Map (Coarse)"):
        """
        Visualise a semantic segmentation map using the coarse planning palette.
        """
        plt.figure(figsize=(10, 10))
        if isinstance(semantic_map, str):
            semantic_map = plt.imread(semantic_map)
        # Map class ids to coarse group ids
        coarse_map = CLASS_TO_GROUP[semantic_map]
        color_map = GROUP_PALETTE[coarse_map]
        plt.imshow(color_map)
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

# Example usage:

def main(): 

    # initialize visualiser
    fv_visualiser = FVVisualizer()

    from jepa_drive_wm.data.kitti import KITTISequence
    kitti_sequence = KITTISequence(sequence_nr=0)

    image_0 = kitti_sequence.get_image(0)
    depth_0 = kitti_sequence.get_depth(0)
    motion_0_1 = kitti_sequence.get_camera_se3(0, 1)
    semantic_0 = kitti_sequence.get_semantics(0)

    # visualise RGB image
    fv_visualiser.display_rgb_image(image_0, title="KITTI Sequence 0 - Frame 0 RGB Image")

    # visualise depth map
    fv_visualiser.display_depth_map(depth_0, title="KITTI Sequence 0 - Frame 0 Depth Map")

    # print motion matrix
    print("Motion matrix from frame 0 to frame 1:")
    print(motion_0_1)

    # visualise semantic map
    fv_visualiser.display_semantic_map(semantic_0, title="KITTI Sequence 0 - Frame 0 Semantic Map")

    # visualise semantic map (coarse)
    fv_visualiser.display_semantic_course(semantic_0, title="KITTI Sequence 0 - Frame 0 Semantic Map (Coarse)")

if __name__ == "__main__":
    main()