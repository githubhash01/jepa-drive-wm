"""
Central path configuration — the ONLY place that knows where data lives.

Every location can be overridden with an environment variable; the defaults are
relative to the repo root, so a fresh clone with the data unpacked into
`<repo>/dataset/KITTI` (see scripts/migrate/) works with zero configuration.

On the laptop, `<repo>/dataset/KITTI` is a symlink to the real dataset under
~/Desktop/Datasets/KITTI, so the same defaults work there too.

    JEPA_KITTI_ROOT      KITTI odometry root (contains data_odometry_color/, data_odometry_poses/)
    JEPA_VJEPA_REPO      V-JEPA 2 checkout (vendored at <repo>/vjepa2)
    JEPA_VJEPA_CKPT_DIR  directory holding vjepa2_1_*.pt checkpoints
    JEPA_DINOV3_REPO     DINOv3 checkout (vendored at <repo>/dinov3)
    JEPA_DINOV3_CKPT     DINOv3 backbone checkpoint (.pth)
    JEPA_OUTPUTS_DIR     trained checkpoints, PCA files, other artefacts
    JEPA_FS_ROOT         FoundationStereo checkout (optional; only for depth pseudolabel generation)
    JEPA_ORBSLAM2_ROOT   ORB_SLAM2 checkout (optional; only for pose pseudolabel generation)
"""
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

KITTI_ROOT = pathlib.Path(
    os.environ.get("JEPA_KITTI_ROOT", REPO_ROOT / "dataset" / "KITTI")
).expanduser()

VJEPA_REPO = pathlib.Path(
    os.environ.get("JEPA_VJEPA_REPO", REPO_ROOT / "vjepa2")
).expanduser()

VJEPA_CKPT_DIR = pathlib.Path(
    os.environ.get("JEPA_VJEPA_CKPT_DIR", VJEPA_REPO / "model_checkpoints" / "vjepa21")
).expanduser()

DINOV3_REPO = pathlib.Path(
    os.environ.get("JEPA_DINOV3_REPO", REPO_ROOT / "dinov3")
).expanduser()

DINOV3_CKPT = pathlib.Path(
    os.environ.get(
        "JEPA_DINOV3_CKPT",
        DINOV3_REPO / "dinov3_checkpoints" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    )
).expanduser()

OUTPUTS_DIR = pathlib.Path(
    os.environ.get("JEPA_OUTPUTS_DIR", REPO_ROOT / "outputs")
).expanduser()

# FoundationStereo is only needed to (re)generate depth pseudolabels, which is
# already done — so this may legitimately not exist on a new machine.
FS_ROOT = pathlib.Path(
    os.environ.get("JEPA_FS_ROOT", "~/Modules/FoundationStereo")
).expanduser()

# ORB_SLAM2 is only needed to (re)generate pose pseudolabels for the KITTI
# sequences without ground truth (11-21) — may not exist on a new machine.
ORBSLAM2_ROOT = pathlib.Path(
    os.environ.get("JEPA_ORBSLAM2_ROOT", "~/Desktop/ORB_SLAM2")
).expanduser()

# Derived KITTI layout (unchanged from the original download structure).
SEQUENCES_DIR = KITTI_ROOT / "data_odometry_color" / "dataset" / "sequences"
GT_POSES_DIR = KITTI_ROOT / "data_odometry_poses" / "dataset" / "poses"
