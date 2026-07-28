"""
class Visualiser:

responsible for all visualisation of data and results

"""

import numpy as np
import matplotlib.pyplot as plt
from jepa_drive_wm.paths import OUTPUTS_DIR

# TODO - write global PCA myself
from jepa_drive_wm.data.global_pca import (
    load_pca_npz,
    transform_features_with_saved_pca,
    normalize_pca_rgb,
)


# --------------------------------------------------------------- Cityscapes-19 taxonomy
#
# The OneFormer label generator writes class ids 0..18 (Cityscapes train ids). This block
# holds the class names + display colours and the coarse 19->5 planning grouping, shared
# by the label generator (data prep) and the semantics eval/viz.

# 19 Cityscapes classes (train ids 0..18) with fixed, human-readable colours (driving-similar
# classes pushed far apart in hue).
CITYSCAPES = [
    ("road",          (128,  64, 128)),  # 0
    ("sidewalk",      (255, 128,   0)),  # 1
    ("building",      (110, 110, 110)),  # 2
    ("wall",          (166, 100,  40)),  # 3
    ("fence",         (190, 153, 153)),  # 4
    ("pole",          (230, 230, 230)),  # 5
    ("traffic light", (250, 170,  30)),  # 6
    ("traffic sign",  (220, 220,   0)),  # 7
    ("vegetation",    ( 60, 160,  40)),  # 8
    ("terrain",       (152, 251, 152)),  # 9
    ("sky",           ( 70, 180, 250)),  # 10
    ("person",        (255,   0,   0)),  # 11
    ("rider",         (255,   0, 200)),  # 12
    ("car",           (  0,   0, 230)),  # 13
    ("truck",         (  0, 128, 255)),  # 14
    ("bus",           (140,   0, 255)),  # 15
    ("train",         (  0,  80, 100)),  # 16
    ("motorcycle",    (255, 200,   0)),  # 17
    ("bicycle",       (119,  11,  32)),  # 18
]
CLASS_NAMES = [name for name, _ in CITYSCAPES]
PALETTE = np.array([c for _, c in CITYSCAPES], dtype=np.uint8)
NUM_CLASSES = len(CITYSCAPES)

# Coarse planning-relevant groups. Each fine class maps to exactly one.
GROUPS = [
    ("drivable",        ( 80, 200,  80)),  # 0
    ("soft-drivable",   (200, 230, 120)),  # 1
    ("static obstacle", (140, 140, 140)),  # 2
    ("dynamic object",  (230,  30,  30)),  # 3
    ("sky / ignore",    (120, 190, 240)),  # 4
]
GROUP_NAMES = [name for name, _ in GROUPS]
GROUP_PALETTE = np.array([c for _, c in GROUPS], dtype=np.uint8)
NUM_GROUPS = len(GROUPS)

_DRIVABLE, _SOFT, _STATIC, _DYNAMIC, _SKY = 0, 1, 2, 3, 4

# Fine Cityscapes class id (0..18) -> coarse planning group id.
CLASS_TO_GROUP = np.array([
    _DRIVABLE,  # 0  road
    _SOFT,      # 1  sidewalk
    _STATIC,    # 2  building
    _STATIC,    # 3  wall
    _STATIC,    # 4  fence
    _STATIC,    # 5  pole
    _STATIC,    # 6  traffic light
    _STATIC,    # 7  traffic sign
    _STATIC,    # 8  vegetation
    _SOFT,      # 9  terrain
    _SKY,       # 10 sky
    _DYNAMIC,   # 11 person
    _DYNAMIC,   # 12 rider
    _DYNAMIC,   # 13 car
    _DYNAMIC,   # 14 truck
    _DYNAMIC,   # 15 bus
    _DYNAMIC,   # 16 train
    _DYNAMIC,   # 17 motorcycle
    _DYNAMIC,   # 18 bicycle
], dtype=np.int64)


def labels_to_groups(label_map: np.ndarray) -> np.ndarray:
    """Map an (H, W) array of Cityscapes class ids to coarse planning group ids."""
    return CLASS_TO_GROUP[label_map]


def class_colors(label_map: np.ndarray) -> np.ndarray:
    """(H, W) Cityscapes ids -> (H, W, 3) RGB. Ids outside 0..18 (255 = unlabeled) render black."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:NUM_CLASSES] = PALETTE
    return lut[label_map]


def group_colors(label_map: np.ndarray) -> np.ndarray:
    """(H, W) Cityscapes ids -> (H, W, 3) coarse-group RGB. Unlabeled renders black."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:NUM_CLASSES] = GROUP_PALETTE[CLASS_TO_GROUP]
    return lut[label_map]

class Visualiser:

    def __init__(self, pca_path: str | None = None) -> None:
        if pca_path is None:
            pca_path = str(OUTPUTS_DIR / "vjepa21_global_train_pca.npz")
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
        color_map = class_colors(semantic_map)
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
        color_map = group_colors(semantic_map)
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