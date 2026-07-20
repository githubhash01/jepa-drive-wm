#!/usr/bin/env bash
# Unpack the USB payload into a fresh clone on the university machine.
# Run from anywhere; give it the stick's transfer folder and the repo root:
#
#   ./scripts/migrate/unpack_uni.sh /media/<user>/<STICK>/jepa-drive-wm-transfer /path/to/jepa-drive-wm
#
# Lays the data out exactly where the code's defaults expect it:
#   <repo>/dataset/KITTI/data_odometry_color/dataset/sequences/...
#   <repo>/dataset/KITTI/data_odometry_poses/dataset/poses/...
#   <repo>/vjepa2/model_checkpoints/vjepa21/vjepa2_1_vitb_dist_vitG_384.pt
set -euo pipefail

SRC="${1:?usage: unpack_uni.sh /path/to/usb/jepa-drive-wm-transfer /path/to/repo}"
REPO="${2:?usage: unpack_uni.sh /path/to/usb/jepa-drive-wm-transfer /path/to/repo}"

mkdir -p "$REPO/dataset/KITTI" "$REPO/vjepa2/model_checkpoints/vjepa21"

rsync -a --info=progress2 "$SRC/KITTI/" "$REPO/dataset/KITTI/"
rsync -a --info=progress2 "$SRC/checkpoints/" "$REPO/vjepa2/model_checkpoints/vjepa21/"

echo
echo "Unpacked. Now regenerate the V-JEPA latents:"
echo "  ./scripts/migrate/build_latents_all.sh"
