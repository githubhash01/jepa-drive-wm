"""
One-off: consolidate KITTI semantics into each sequence folder.

Moves   .../semantic_oneformer/XX/*.png
into    .../data_odometry_color/dataset/sequences/XX/semantic_oneformer/*.png

so every sequence folder holds its RGB (image_2/3), depth, vjepa_vitb latents,
AND semantics side by side. Idempotent: re-running skips sequences already moved.

    python scripts/reorg_semantics.py            # dry run, prints what it would do
    python scripts/reorg_semantics.py --apply    # actually move the files
"""
import argparse
import pathlib
import shutil

SEQUENCES_DIR = pathlib.Path(
    "/home/hashim/Desktop/Datasets/KITTI/data_odometry_color/dataset/sequences"
)
SEMANTICS_SRC = pathlib.Path("/home/hashim/Desktop/Datasets/KITTI/semantic_oneformer")
SEMANTICS_DIRNAME = "semantic_oneformer"  # subdir name created inside each sequence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move (default: dry run)")
    args = ap.parse_args()

    for src_seq in sorted(SEMANTICS_SRC.iterdir()):
        if not src_seq.is_dir():
            continue
        seq = src_seq.name  # "00", "01", ...
        dst_seq = SEQUENCES_DIR / seq / SEMANTICS_DIRNAME

        if not (SEQUENCES_DIR / seq).exists():
            print(f"[skip] {seq}: no target sequence folder {SEQUENCES_DIR / seq}")
            continue
        if dst_seq.exists():
            print(f"[skip] {seq}: already present at {dst_seq}")
            continue

        pngs = sorted(src_seq.glob("*.png"))
        print(f"[move] {seq}: {len(pngs)} files  {src_seq}  ->  {dst_seq}")
        if not args.apply:
            continue

        dst_seq.mkdir(parents=True, exist_ok=True)
        for p in pngs:
            shutil.move(str(p), str(dst_seq / p.name))
        # remove the now-empty source folder if nothing else is left in it
        if not any(src_seq.iterdir()):
            src_seq.rmdir()

    if not args.apply:
        print("\nDry run. Re-run with --apply to move the files.")


if __name__ == "__main__":
    main()
