"""Cityscapes-19 taxonomy for the semantic-segmentation task.

The OneFormer label generator writes class ids 0..18 (Cityscapes train ids). This module holds
the class names + display colours and the coarse 19->5 planning grouping, shared by the label
generator (data prep) and the semseg task (metrics/viz).
"""
from __future__ import annotations

import numpy as np

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
# GROUP_PALETTE = np.array([c for _, c in GROUPS], dtype=np.uint8)

# Make the colours more visually distinct: 

# drivable: dark grey 
# soft-drivable: light grey 
# static obstacle: dark blue
# dynamic object: dark red
# sky / ignore: light blue
GROUP_PALETTE = np.array([
    ( 80,  80,  80),   # drivable
    (200, 200, 200),   # soft-drivable
    (  0,   0, 150),   # static obstacle
    (150,   0,   0),   # dynamic object
    (150, 200, 255),   # sky / ignore
], dtype=np.uint8)

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
