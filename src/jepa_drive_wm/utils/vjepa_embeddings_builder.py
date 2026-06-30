#!/usr/bin/env python3
"""
Use the smallest VJEPA model and embed every single frame in our KITTI dataset.

This mirrors depth_builder_allseq.py, but instead of stereo depth it caches a
per-frame V-JEPA 2.1 embedding next to the images/depth that already live inside
each KITTI sequence:

    .../sequences/00/
        image_2/   000000.png ...
        image_3/   000000.png ...
        depth/     000000.png ...
        vjepa_vitb/                 <- created here
            000000.npy   (num_tokens, embed_dim) token grid for that frame
            000001.npy
            ...
            _metadata.json          wrapper provenance (size, ckpt, image size...)

Each frame is encoded *independently* in image mode (T=1), so the cache is just
one small array per image and can be rebuilt incrementally (existing files are
skipped unless --overwrite).

Run with PYTHONPATH=src, e.g.:
    PYTHONPATH=src python -m jepa_drive_wm.utils.vjepa_embeddings_builder --sequences 0-21
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vjepa_wrapper import VJEPA21Size, VJEPA21Wrapper


def parse_sequences(spec: str) -> list[int]:
    """Parse comma/range sequence specs like '0-21', '00,02,10', or '0-10,13'."""
    out: list[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    # preserve order, remove duplicates
    seen = set()
    seqs = []
    for s in out:
        if s < 0 or s > 21:
            raise ValueError(f"KITTI odometry sequence out of expected range 00-21: {s}")
        if s not in seen:
            seen.add(s)
            seqs.append(s)
    if not seqs:
        raise ValueError(f"No sequences parsed from: {spec!r}")
    return seqs


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", type=int, default=0, help="Single sequence if --sequences is not set.")
    ap.add_argument("--sequences", type=str, default=None, help="Comma/range list, e.g. '0-21' or '00,02,10'.")
    ap.add_argument("--kitti_root", default="/home/hashim/Desktop/Datasets/KITTI")
    ap.add_argument("--image_dirname", default="image_2", help="Which camera folder to embed.")
    ap.add_argument("--out_dirname", default=None,
                    help="Cache folder created inside each sequence. Defaults to "
                         "'vjepa_vitb' (final layer) or 'vjepa_vitb_hier' with --hierarchical.")
    ap.add_argument("--hierarchical", action="store_true",
                    help="Cache 4 intermediate layers ([2,5,8,11] for ViT-B) as (L, num_tokens, D) "
                         "per frame instead of the final layer only.")
    ap.add_argument("--image_height", type=int, default=384)
    ap.add_argument("--image_width", type=int, default=1248)
    ap.add_argument("--batch_size", type=int, default=8, help="Frames per encoder pass; lower if you hit OOM.")
    ap.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16", help="On-disk dtype for the .npy cache.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_frames", type=int, default=0, help="For quick benchmarking; 0 means all frames.")
    args = ap.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    if args.out_dirname is None:
        args.out_dirname = "vjepa_vitb_hier" if args.hierarchical else "vjepa_vitb"

    np_dtype = np.float16 if args.save_dtype == "fp16" else np.float32

    sequences = parse_sequences(args.sequences) if args.sequences is not None else [args.sequence]
    print(f"Sequences: {', '.join(f'{s:02d}' for s in sequences)}")

    # Smallest (and fastest) V-JEPA 2.1 checkpoint, built once and reused.
    wrapper = VJEPA21Wrapper(
        size=VJEPA21Size.BASE,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    print(wrapper)

    total_start = time.perf_counter()
    total_done = 0
    total_skipped_existing = 0

    for seq in sequences:
        seq_dir = Path(args.kitti_root) / f"data_odometry_color/dataset/sequences/{seq:02d}"
        img_dir = seq_dir / args.image_dirname
        out_dir = seq_dir / args.out_dirname
        out_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.is_dir():
            raise FileNotFoundError(f"{img_dir} not found")

        frame_ids = sorted(p.stem for p in img_dir.glob("*.png"))
        if args.max_frames > 0:
            frame_ids = frame_ids[: args.max_frames]
        if not frame_ids:
            raise FileNotFoundError(f"No frames found in {img_dir}")

        # Provenance, written once per sequence cache.
        meta = wrapper.metadata()
        meta["layout"] = vars(wrapper.layout(num_frames=1))
        meta["save_dtype"] = args.save_dtype
        meta["hierarchical"] = args.hierarchical
        if args.hierarchical:
            meta["out_layers"] = wrapper.hierarchical_layers
            meta["n_layers"] = len(wrapper.hierarchical_layers)
        (out_dir / "_metadata.json").write_text(json.dumps(meta, indent=2))

        work = []
        skipped_existing = 0
        for fid in frame_ids:
            out_path = out_dir / f"{fid}.npy"
            if out_path.exists() and not args.overwrite:
                skipped_existing += 1
                continue
            work.append((fid, img_dir / f"{fid}.png", out_path))

        total_skipped_existing += skipped_existing
        print(f"\nSequence {seq:02d}: {len(frame_ids)} frames -> {out_dir}")
        print(f"Frames to process: {len(work)}  already existed: {skipped_existing}")
        if not work:
            continue

        seq_start = time.perf_counter()
        done = 0

        for batch_items in chunks(work, args.batch_size):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            paths = [p for _, p, _ in batch_items]
            if args.hierarchical:
                feats = wrapper.encode_images_hierarchical(paths, batch_size=len(paths))  # (B, L, num_tokens, D)
            else:
                feats = wrapper.encode_images(paths, batch_size=len(paths))  # (B, num_tokens, D) on cpu

            for emb, (_, _, out_path) in zip(feats, batch_items):
                np.save(out_path, emb.numpy().astype(np_dtype, copy=False))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            done += len(batch_items)
            total_done += len(batch_items)

            if done == len(batch_items) or done % 100 == 0 or done >= len(work):
                seq_elapsed = time.perf_counter() - seq_start
                total_elapsed = time.perf_counter() - total_start
                print(
                    f"  seq{seq:02d}/{batch_items[-1][0]}  batch {dt:.2f}s, {dt/len(batch_items):.3f}s/frame "
                    f"({done}/{len(work)} seq, {total_done} total, "
                    f"seq avg {seq_elapsed/max(done,1):.3f}s/frame, total avg {total_elapsed/max(total_done,1):.3f}s/frame)"
                )

        seq_elapsed = time.perf_counter() - seq_start
        print(f"Sequence {seq:02d} finished {done} frames in {seq_elapsed:.1f}s ({seq_elapsed/max(done,1):.3f}s/frame)")

    total_elapsed = time.perf_counter() - total_start
    print(
        f"\nFinished all requested sequences: wrote {total_done} frames, "
        f"skipped existing {total_skipped_existing}, elapsed {total_elapsed:.1f}s "
        f"({total_elapsed/max(total_done,1):.3f}s/frame over written frames)"
    )


if __name__ == "__main__":
    main()
