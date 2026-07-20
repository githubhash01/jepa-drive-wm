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
    JEPA_OUTPUTS_DIR     trained checkpoints, PCA files, other artefacts
    JEPA_FS_ROOT         FoundationStereo checkout (optional; only for depth pseudolabel generation)
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

OUTPUTS_DIR = pathlib.Path(
    os.environ.get("JEPA_OUTPUTS_DIR", REPO_ROOT / "outputs")
).expanduser()

# FoundationStereo is only needed to (re)generate depth pseudolabels, which is
# already done — so this may legitimately not exist on a new machine.
FS_ROOT = pathlib.Path(
    os.environ.get("JEPA_FS_ROOT", "~/Modules/FoundationStereo")
).expanduser()

# Derived KITTI layout (unchanged from the original download structure).
SEQUENCES_DIR = KITTI_ROOT / "data_odometry_color" / "dataset" / "sequences"
GT_POSES_DIR = KITTI_ROOT / "data_odometry_poses" / "dataset" / "poses"
