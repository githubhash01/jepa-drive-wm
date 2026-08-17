# TODO - write this much simpler myself 

#!/usr/bin/env python3
"""



global_pca_kitti.py

Fit one global PCA basis over a sampled subset of cached V-JEPA patch tokens,
then use that same PCA basis to visualize any KITTI frame.

Why this exists:
    Per-image PCA gives pretty pictures, but colours are not comparable across
    frames/sequences because each image gets its own PCA coordinate system.
    This script fits PCA once over sampled training patches, saves the basis,
    and applies the same basis everywhere.

Typical use (from the repo root, PYTHONPATH=src):

    python -m jepa_drive_wm.data.global_pca --fit --visualize

Useful variants:

    # fit from more patches, but cap frames per sequence
    python -m jepa_drive_wm.data.global_pca --fit --patches-per-frame 256 --max-frames-per-seq 1000

    # only visualize using an existing saved PCA
    python -m jepa_drive_wm.data.global_pca --visualize --viz-sequences 0 1 2 3 4 5 6 7 9 10 --viz-frame 0

    # make smoother image-sized overlays as well as honest patch-grid images
    python -m jepa_drive_wm.data.global_pca --visualize --upsample bilinear
"""

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from PIL import Image
import matplotlib.pyplot as plt
from jepa_drive_wm.data.splits import SPLIT_V1
from jepa_drive_wm.paths import OUTPUTS_DIR
OUT_DIR = str(OUTPUTS_DIR)


# The PCA basis is a training-set statistic: fit only on the train split so no
# val/test data leaks into it. Visualising with the frozen basis is harmless on
# any sequence.
DEFAULT_FIT_SEQUENCES = list(SPLIT_V1.train_sequences)
DEFAULT_VIZ_SEQUENCES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]

@dataclass 
class ImageEmbedding: 
    image_path: str # path to the original image
    original_image_dimensions: tuple[int, int] # (height, width)
    processed_image_dimensions: tuple[int, int] # (height, width) after resizing
    grid_h: int
    grid_w: int
    patch_size: int
    features: torch.Tensor # (GRID_H * GRID_W, feature_dim)
    model_name: str
    checkpoint_path: str
    encoder_key: str = "ema_encoder"

@dataclass
class DataPoint:
    sequence_nr: int
    model_name: str

    timestamp: float
    frame_idx: int
    image_embedding: ImageEmbedding


@dataclass
class EmbeddingRecord:
    seq: int
    frame: int
    path: str


def discover_embedding_records(sequences, model="vjepa21_large", out_dir=OUT_DIR):
    records = []
    for seq in sequences:
        emb_dir = os.path.join(out_dir, f"seq{seq:02d}_{model}")
        paths = sorted(glob.glob(os.path.join(emb_dir, "*.pt")))
        if len(paths) == 0:
            print(f"[warn] no embeddings found for seq {seq:02d}: {emb_dir}")
            continue

        for path in paths:
            frame = int(os.path.basename(path)[:-3])
            records.append(EmbeddingRecord(seq=seq, frame=frame, path=path))

    if len(records) == 0:
        raise RuntimeError("No embedding files found. Check OUT_DIR, model name, and sequences.")
    return records


def optionally_limit_frames_per_sequence(records, max_frames_per_seq=None, seed=0):
    if max_frames_per_seq is None:
        return records

    rng = np.random.default_rng(seed)
    out = []
    seqs = sorted(set(r.seq for r in records))

    for seq in seqs:
        seq_records = [r for r in records if r.seq == seq]
        if len(seq_records) > max_frames_per_seq:
            idx = rng.choice(len(seq_records), size=max_frames_per_seq, replace=False)
            seq_records = [seq_records[i] for i in sorted(idx)]
        out.extend(seq_records)

    return out


def load_embedding(record: EmbeddingRecord):
    dp: DataPoint = torch.load(record.path, weights_only=False)
    return dp.image_embedding


def sample_patch_features(features: torch.Tensor, patches_per_frame: int, rng: np.random.Generator):
    """
    features: (num_patches, D), usually bf16/float tensor on CPU.
    returns: numpy float32 array, (K, D)
    """
    n = features.shape[0]
    k = min(patches_per_frame, n)
    idx = rng.choice(n, size=k, replace=False)
    return features[idx].float().cpu().numpy().astype(np.float32, copy=False)


def flush_partial_fit(ipca: IncrementalPCA, buffer, n_components: int):
    if len(buffer) == 0:
        return 0

    X = np.concatenate(buffer, axis=0)
    if X.shape[0] < n_components:
        return 0

    ipca.partial_fit(X)
    return X.shape[0]


def fit_incremental_pca(
    records,
    n_components=3,
    patches_per_frame=128,
    batch_patches=16384,
    seed=0,
):
    """
    Iteratively load frames, sample patches, and partial_fit IncrementalPCA.
    """
    rng = np.random.default_rng(seed)
    ipca = IncrementalPCA(n_components=n_components)

    buffer = []
    buffered = 0
    total_patches = 0
    first_shape = None

    pbar = tqdm(records, desc="fitting global PCA", dynamic_ncols=True)
    for record in pbar:
        emb = load_embedding(record)
        X = sample_patch_features(emb.features, patches_per_frame, rng)

        if first_shape is None:
            first_shape = tuple(emb.features.shape)
            print(f"first embedding shape: {first_shape}")
            print(f"grid: {emb.grid_h} x {emb.grid_w}, patch size: {emb.patch_size}")

        buffer.append(X)
        buffered += X.shape[0]

        if buffered >= batch_patches:
            n_fit = flush_partial_fit(ipca, buffer, n_components)
            total_patches += n_fit
            buffer = []
            buffered = 0
            pbar.set_postfix(patches=total_patches)

    if buffered > 0:
        n_fit = flush_partial_fit(ipca, buffer, n_components)
        total_patches += n_fit

    print(f"fitted PCA on approximately {total_patches:,} sampled patches")
    print("explained variance ratio:", ipca.explained_variance_ratio_)
    print("sum explained variance:", float(ipca.explained_variance_ratio_.sum()))
    return ipca


def estimate_rgb_percentiles(
    ipca,
    records,
    patches_per_frame=128,
    max_records=1000,
    seed=123,
    low_pct=1.0,
    high_pct=99.0,
):
    """
    Estimate global PCA-channel normalization using sampled transformed patches.
    This keeps colours comparable across frames.
    """
    rng = np.random.default_rng(seed)
    if len(records) > max_records:
        idx = rng.choice(len(records), size=max_records, replace=False)
        records = [records[i] for i in sorted(idx)]

    projected = []
    for record in tqdm(records, desc="estimating PCA colour scale", dynamic_ncols=True):
        emb = load_embedding(record)
        X = sample_patch_features(emb.features, patches_per_frame, rng)
        Y = ipca.transform(X).astype(np.float32, copy=False)
        projected.append(Y)

    Y = np.concatenate(projected, axis=0)
    lo = np.percentile(Y, low_pct, axis=0)
    hi = np.percentile(Y, high_pct, axis=0)

    # Prevent divide-by-zero if a component is degenerate.
    hi = np.where(hi > lo, hi, lo + 1e-6)

    print(f"PCA RGB percentile low  ({low_pct}%):", lo)
    print(f"PCA RGB percentile high ({high_pct}%):", hi)
    return lo.astype(np.float32), hi.astype(np.float32)


def save_pca_npz(ipca, rgb_low, rgb_high, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        mean=ipca.mean_.astype(np.float32),
        components=ipca.components_.astype(np.float32),
        explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32),
        rgb_low=rgb_low.astype(np.float32),
        rgb_high=rgb_high.astype(np.float32),
    )
    print("saved global PCA to:", save_path)


def load_pca_npz(path):
    data = np.load(path)
    return {
        "mean": data["mean"].astype(np.float32),
        "components": data["components"].astype(np.float32),
        "explained_variance_ratio": data["explained_variance_ratio"].astype(np.float32),
        "rgb_low": data["rgb_low"].astype(np.float32),
        "rgb_high": data["rgb_high"].astype(np.float32),
    }


def transform_features_with_saved_pca(features: torch.Tensor, pca_data):
    X = features.float().cpu().numpy().astype(np.float32, copy=False)
    X_centered = X - pca_data["mean"]
    Y = X_centered @ pca_data["components"].T
    return Y.astype(np.float32, copy=False)


def normalize_pca_rgb(Y, rgb_low, rgb_high):
    """
    Y: (H, W, 3) PCA projection.
    Normalize using global train-set percentiles, not per-image min/max.
    """
    out = (Y - rgb_low.reshape(1, 1, 3)) / (rgb_high - rgb_low).reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def visualize_record(record, pca_data, model="vjepa21_large", upsample="none", out_dir=OUT_DIR):
    emb = load_embedding(record)
    grid_h, grid_w = emb.grid_h, emb.grid_w

    Y = transform_features_with_saved_pca(emb.features, pca_data)
    pca_img = Y.reshape(grid_h, grid_w, 3)
    pca_rgb = normalize_pca_rgb(pca_img, pca_data["rgb_low"], pca_data["rgb_high"])

    original = None
    if emb.image_path is not None and os.path.exists(emb.image_path):
        original = Image.open(emb.image_path).convert("RGB")

    if upsample != "none":
        # This is still patch-token PCA, just interpolated for display.
        pil = Image.fromarray((pca_rgb * 255).astype(np.uint8))
        target_size = (emb.processed_image_dimensions[1], emb.processed_image_dimensions[0])  # PIL wants W,H

        if upsample == "nearest":
            resample = Image.Resampling.NEAREST
        elif upsample == "bilinear":
            resample = Image.Resampling.BILINEAR
        else: 
            resample = Image.Resampling.BICUBIC

        pca_display = np.asarray(pil.resize(target_size, resample=resample)).astype(np.float32) / 255.0
        title_extra = f"upsampled {upsample}"
    else:
        pca_display = pca_rgb
        title_extra = f"{grid_h}x{grid_w} patches"

    if original is not None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].imshow(original)
        axes[0].set_title(f"Original KITTI seq {record.seq:02d} frame {record.frame:06d}")
        axes[0].axis("off")

        axes[1].imshow(pca_display, interpolation="nearest", aspect="equal")
        axes[1].set_title(f"Global V-JEPA PCA ({title_extra})")
        axes[1].axis("off")
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 3))
        axes.imshow(pca_display, interpolation="nearest", aspect="equal")
        axes.set_title(f"Global V-JEPA PCA seq {record.seq:02d} frame {record.frame:06d} ({title_extra})")
        axes.axis("off")

    save_name = f"seq{record.seq:02d}_frame{record.frame:06d}_global_pca.png"
    save_path = os.path.join(out_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print("saved visualization:", save_path)


def find_record(records, seq, frame):
    for r in records:
        if r.seq == seq and r.frame == frame:
            return r
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true", help="fit and save global PCA")
    parser.add_argument("--visualize", action="store_true", help="visualize frames using saved global PCA")

    parser.add_argument("--model", type=str, default="vjepa21_large")
    parser.add_argument("--pca-path", type=str, default=os.path.join(OUT_DIR, "vjepa21_global_train_pca.npz"))

    parser.add_argument("--fit-sequences", type=int, nargs="+", default=DEFAULT_FIT_SEQUENCES)
    parser.add_argument("--viz-sequences", type=int, nargs="+", default=DEFAULT_VIZ_SEQUENCES)
    parser.add_argument("--viz-frame", type=int, default=0)

    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--patches-per-frame", type=int, default=128)
    parser.add_argument("--batch-patches", type=int, default=16384)
    parser.add_argument("--max-frames-per-seq", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--scale-patches-per-frame", type=int, default=128)
    parser.add_argument("--scale-max-records", type=int, default=1000)
    parser.add_argument("--low-pct", type=float, default=1.0)
    parser.add_argument("--high-pct", type=float, default=99.0)

    parser.add_argument(
        "--upsample",
        type=str,
        default="none",
        choices=["none", "nearest", "bilinear", "bicubic"],
        help="optional display-only upsampling of patch PCA to image size",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.fit and not args.visualize:
        print("Nothing to do. Use --fit, --visualize, or both.")
        return

    if args.fit:
        fit_records = discover_embedding_records(args.fit_sequences, model=args.model)
        fit_records = optionally_limit_frames_per_sequence(
            fit_records,
            max_frames_per_seq=args.max_frames_per_seq,
            seed=args.seed,
        )
        print(f"fit sequences: {args.fit_sequences}")
        print(f"fit records:   {len(fit_records)}")
        print(f"patches/frame: {args.patches_per_frame}")

        ipca = fit_incremental_pca(
            fit_records,
            n_components=args.n_components,
            patches_per_frame=args.patches_per_frame,
            batch_patches=args.batch_patches,
            seed=args.seed,
        )

        rgb_low, rgb_high = estimate_rgb_percentiles(
            ipca,
            fit_records,
            patches_per_frame=args.scale_patches_per_frame,
            max_records=args.scale_max_records,
            seed=args.seed + 123,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
        )

        save_pca_npz(ipca, rgb_low, rgb_high, args.pca_path)

    if args.visualize:
        if not os.path.exists(args.pca_path):
            raise FileNotFoundError(f"Missing PCA file: {args.pca_path}. Run with --fit first.")

        pca_data = load_pca_npz(args.pca_path)
        print("loaded PCA:", args.pca_path)
        print("explained variance ratio:", pca_data["explained_variance_ratio"])
        print("sum explained variance:", float(pca_data["explained_variance_ratio"].sum()))

        viz_records_all = discover_embedding_records(args.viz_sequences, model=args.model)
        for seq in args.viz_sequences:
            record = find_record(viz_records_all, seq=seq, frame=args.viz_frame)
            if record is None:
                print(f"[warn] missing seq {seq:02d} frame {args.viz_frame:06d}")
                continue
            visualize_record(record, pca_data, model=args.model, upsample=args.upsample)


if __name__ == "__main__":
    main()