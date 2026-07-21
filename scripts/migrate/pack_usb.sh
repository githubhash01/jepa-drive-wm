#!/usr/bin/env bash
# Pack everything the university machine needs onto a USB stick (~41G).
#
#   ./scripts/migrate/pack_usb.sh /media/hashim/<STICK>
#
# What goes on the stick (and why):
#   KITTI/sequences/<seq>/{image_2,depth,semantic_oneformer,calib.txt,times.txt,K.txt}
#       image_2 is only needed to REGENERATE the V-JEPA latents on the GPU;
#       depth + semantics are homemade pseudolabels (not downloadable).
#   KITTI/poses/                         ground-truth odometry poses (3.6M)
#   checkpoints/vjepa2_1_vitb_dist_vitG_384.pt   the only checkpoint the code uses (1.6G)
#
# Deliberately excluded: image_3 (32G, only used to build depth — already done),
# vjepa_vitb latents (117G, regenerated on the GPU), ViT-L/ViT-G checkpoints (21G).
#
# rsync is resumable: re-run after an interruption and it picks up where it left off.
set -euo pipefail

USB_ROOT="${1:?usage: pack_usb.sh /path/to/usb/mount}"
KITTI_ROOT="${JEPA_KITTI_ROOT:-$HOME/Desktop/Datasets/KITTI}"
CKPT="$(dirname "$0")/../../vjepa2/model_checkpoints/vjepa21/vjepa2_1_vitb_dist_vitG_384.pt"

SEQ_SRC="$KITTI_ROOT/data_odometry_color/dataset/sequences"
POSES_SRC="$KITTI_ROOT/data_odometry_poses"
DEST="$USB_ROOT/jepa-drive-wm-transfer"

echo "Packing to $DEST"
# rsync only creates the final component of its destination, not the whole
# chain, so pre-create the nested KITTI dirs (and checkpoints).
mkdir -p "$DEST/checkpoints" \
         "$DEST/KITTI/data_odometry_color/dataset/sequences" \
         "$DEST/KITTI/data_odometry_poses"

# -rt (not -a): keep it recursive + timestamps (so re-runs skip done files) but
# DON'T try to preserve Unix perms/ownership -- exFAT/FAT can't store them and
# rsync would exit non-zero under `set -e`. --partial keeps half-copied files
# (e.g. the 1.6G checkpoint) so an interrupted run resumes instead of restarting.
RSYNC="rsync -rt --partial --info=progress2"

# Sequences: everything except image_3 and the regenerable latent caches.
$RSYNC \
    --exclude='image_3/' \
    --exclude='vjepa_vitb/' \
    --exclude='vjepa_vitb_hier/' \
    "$SEQ_SRC/" "$DEST/KITTI/data_odometry_color/dataset/sequences/"

# Ground-truth poses (tiny).
$RSYNC "$POSES_SRC/" "$DEST/KITTI/data_odometry_poses/"

# The one checkpoint the code loads.
$RSYNC "$CKPT" "$DEST/checkpoints/"

echo
echo "Done. Stick payload:"
du -sh "$DEST"/*
