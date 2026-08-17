#!/usr/bin/env python3
"""
Normalise the KITTI depth maps to one uniform, full-resolution layout.

The depth dir was a mess: some sequences were exported at native (full) resolution,
most at scale 0.5, plus stray ``depth_fast`` dirs. The half-res depths are already
*metrically correct* (the disparity->depth conversion accounts for scale), so we do
NOT need to re-run FoundationStereo -- we just upsample each depth map to its own
sequence's native ``image_2`` resolution.

For every sequence this script:
  * reads each PNG in ``depth/`` (uint16, KITTI value = depth_m * 256),
  * nearest-upsamples it to the native image resolution if it is smaller
    (nearest preserves exact metric values and crisp invalid=0 regions),
  * atomically overwrites the PNG in place (write tmp -> os.replace),
  * leaves already-native and 0-byte/corrupt frames untouched (reported),
  * deletes the stray ``depth_fast/`` dir at the end.

After this, ``depth/`` for each sequence matches that sequence's ``image_2`` size,
so the probe can train/val across any split with depth stored at native resolution.

Run (dry-run first is wise):
    PYTHONPATH=src python -m jepa_drive_wm.data.dataset_builders.normalize_kitti_depth --dry_run
    PYTHONPATH=src python -m jepa_drive_wm.data.dataset_builders.normalize_kitti_depth
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

from jepa_drive_wm.paths import KITTI_ROOT

import cv2


def parse_sequences(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    seen, seqs = set(), []
    for s in out:
        if not 0 <= s <= 21:
            raise ValueError(f"KITTI odometry sequence out of range 00-21: {s}")
        if s not in seen:
            seen.add(s)
            seqs.append(s)
    return seqs


def native_image_size(seq_dir: Path) -> tuple[int, int] | None:
    """(H, W) of the first image_2 frame, or None if there are no images."""
    imgs = sorted(glob.glob(str(seq_dir / "image_2" / "*.png")))
    if not imgs:
        return None
    img = cv2.imread(imgs[0], cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.shape[0], img.shape[1]


def normalize_sequence(seq_dir: Path, depth_dirname: str, extra_dirnames: list[str], dry_run: bool):
    seq = seq_dir.name
    depth_dir = seq_dir / depth_dirname

    size = native_image_size(seq_dir)
    if size is None:
        print(f"seq {seq}: no images, skipping")
        return
    th, tw = size

    if not depth_dir.is_dir():
        print(f"seq {seq}: no {depth_dirname}/ dir, skipping")
        return

    pngs = sorted(glob.glob(str(depth_dir / "*.png")))
    upsampled = already_native = corrupt = 0
    src_hw: tuple[int, int] | None = None

    for p in pngs:
        raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if raw is None:
            corrupt += 1
            continue
        h, w = raw.shape[:2]
        if src_hw is None:
            src_hw = (h, w)
        if (h, w) == (th, tw):
            already_native += 1
            continue
        upsampled += 1
        if dry_run:
            continue
        up = cv2.resize(raw, (tw, th), interpolation=cv2.INTER_NEAREST)
        tmp = p + ".tmp.png"
        # png_compression 1: fast write, lossless (depth must stay exact).
        cv2.imwrite(tmp, up, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        os.replace(tmp, p)

    src = f"{src_hw[0]}x{src_hw[1]}" if src_hw else "?"
    corrupt_note = f"  corrupt/0-byte: {corrupt}" if corrupt else ""
    action = "would upsample" if dry_run else "upsampled"
    print(
        f"seq {seq}: native {th}x{tw}  src {src}  frames {len(pngs)}  "
        f"{action} {upsampled}, already-native {already_native}{corrupt_note}"
    )

    # Remove stray depth dirs (e.g. depth_fast) once the canonical dir is handled.
    for extra in extra_dirnames:
        d = seq_dir / extra
        if d.is_dir():
            n = len(glob.glob(str(d / "*.png")))
            if dry_run:
                print(f"          would remove {extra}/ ({n} pngs)")
            else:
                shutil.rmtree(d)
                print(f"          removed {extra}/ ({n} pngs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", default=str(KITTI_ROOT))
    ap.add_argument("--sequences", default="0-21", help="Comma/range list, e.g. '0-21' or '00,02,10'.")
    ap.add_argument("--depth_dirname", default="depth", help="Canonical depth dir to normalise in place.")
    ap.add_argument("--remove_dirnames", default="depth_fast", help="Comma list of stray dirs to delete.")
    ap.add_argument("--dry_run", action="store_true", help="Report the plan without writing/deleting anything.")
    args = ap.parse_args()

    sequences = parse_sequences(args.sequences)
    extra = [d.strip() for d in args.remove_dirnames.split(",") if d.strip()]
    seq_root = Path(args.kitti_root) / "data_odometry_color/dataset/sequences"

    mode = "DRY-RUN (no changes)" if args.dry_run else "WRITING in place"
    print(f"[normalize_kitti_depth] {mode}  root={seq_root}")
    print(f"  normalising '{args.depth_dirname}/' to native image_2 resolution; removing {extra}\n")

    for s in sequences:
        normalize_sequence(seq_root / f"{s:02d}", args.depth_dirname, extra, args.dry_run)

    print("\nDone." + ("  Re-run without --dry_run to apply." if args.dry_run else ""))


if __name__ == "__main__":
    main()
