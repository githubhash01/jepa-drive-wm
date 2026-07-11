#!/usr/bin/env python3
"""
OneFormer (Cityscapes) semantic segmentation for KITTI.

Entry points:
  build      -- run the model over one KITTI sequence and save the raw
                19-class Cityscapes label maps as PNGs.
  build-all  -- do the same for every sequence 00..21, skipping finished frames.
  viz        -- visualise one frame's saved segmentation, both as the raw 19
                Cityscapes classes and as the coarse planning grouping (the
                grouping is derived on the fly, never saved).

    python -m jepa_drive_wm.probes.semantics.oneformer_kitti build-all
    python -m jepa_drive_wm.probes.semantics.oneformer_kitti build --seq 10
    python -m jepa_drive_wm.probes.semantics.oneformer_kitti viz   --seq 10 --frame 0

Efficiency notes (tuned for an 8 GB RTX 4070):
  * fp16 autocast on the forward pass  -> ~2x faster, ~half the memory
  * cuDNN autotuning (fixed input size) -> faster convs
  * batched forwards with automatic OOM back-off
  * a background thread prefetches/decodes the next batch of images while the
    GPU is busy, so we are not CPU-bound between forwards.
Progress is durable: each frame is written as soon as it is computed, and
already-present frames are skipped, so you can Ctrl-C any time and resume.
"""

import argparse
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Reduce CUDA fragmentation on small (8 GB) cards. Must be set before torch/CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from tqdm import tqdm
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

from jepa_drive_wm.data.kitti import KITTISequence

MODEL_ID = "shi-labs/oneformer_cityscapes_swin_large"
DATASET_ROOT = Path("/home/hashim/Desktop/Datasets/KITTI/semantic_oneformer")

# The Cityscapes processor upsamples inputs to a large resolution by default,
# which OOMs an 8 GB card. KITTI frames are ~1241x376, so cap the shortest edge
# near native. The saved label map is still upsampled back to full KITTI size
# via post-processing `target_sizes`, so output resolution is unaffected.
SHORTEST_EDGE = 384

# Fixed-size KITTI inputs -> let cuDNN pick the fastest kernels.
torch.backends.cudnn.benchmark = True

# --- 19 Cityscapes classes (train ids 0..18) with fixed, human-readable colours.
#     Colours are chosen so classes that matter for driving but look similar
#     (road vs sidewalk vs terrain) are far apart in hue.
CITYSCAPES = [
    ("road",          (128,  64, 128)),  # 0   purple
    ("sidewalk",      (255, 128,   0)),  # 1   orange
    ("building",      (110, 110, 110)),  # 2   grey
    ("wall",          (166, 100,  40)),  # 3   brown
    ("fence",         (190, 153, 153)),  # 4   dusty pink
    ("pole",          (230, 230, 230)),  # 5   near-white
    ("traffic light", (250, 170,  30)),  # 6   amber
    ("traffic sign",  (220, 220,   0)),  # 7   yellow
    ("vegetation",    ( 60, 160,  40)),  # 8   green
    ("terrain",       (152, 251, 152)),  # 9   pale mint
    ("sky",           ( 70, 180, 250)),  # 10  cyan-blue
    ("person",        (255,   0,   0)),  # 11  red
    ("rider",         (255,   0, 200)),  # 12  magenta
    ("car",           (  0,   0, 230)),  # 13  blue
    ("truck",         (  0, 128, 255)),  # 14  light blue
    ("bus",           (140,   0, 255)),  # 15  violet
    ("train",         (  0,  80, 100)),  # 16  dark teal
    ("motorcycle",    (255, 200,   0)),  # 17  gold
    ("bicycle",       (119,  11,  32)),  # 18  maroon
]
CLASS_NAMES = [name for name, _ in CITYSCAPES]
PALETTE = np.array([c for _, c in CITYSCAPES], dtype=np.uint8)

# --- Coarse planning-relevant groups. Each fine class maps to exactly one.
GROUPS = [
    ("drivable",        ( 80, 200,  80)),  # 0  road surface the car drives on  -> green
    ("soft-drivable",   (200, 230, 120)),  # 1  traversable-ish (pavement/grass) -> yellow-green
    ("static obstacle", (140, 140, 140)),  # 2  fixed non-drivable geometry      -> grey
    ("dynamic object",  (230,  30,  30)),  # 3  things that move                 -> red
    ("sky / ignore",    (120, 190, 240)),  # 4  no planning meaning              -> light blue
]
GROUP_NAMES = [name for name, _ in GROUPS]
GROUP_PALETTE = np.array([c for _, c in GROUPS], dtype=np.uint8)

DRIVABLE, SOFT, STATIC, DYNAMIC, SKY = 0, 1, 2, 3, 4

# Fine Cityscapes class id (0..18) -> coarse planning group id.
CLASS_TO_GROUP = np.array([
    DRIVABLE,   # 0  road
    SOFT,       # 1  sidewalk      (pavement)
    STATIC,     # 2  building
    STATIC,     # 3  wall
    STATIC,     # 4  fence
    STATIC,     # 5  pole
    STATIC,     # 6  traffic light
    STATIC,     # 7  traffic sign
    STATIC,     # 8  vegetation
    SOFT,       # 9  terrain       (grass/dirt verge)
    SKY,        # 10 sky
    DYNAMIC,    # 11 person
    DYNAMIC,    # 12 rider
    DYNAMIC,    # 13 car
    DYNAMIC,    # 14 truck
    DYNAMIC,    # 15 bus
    DYNAMIC,    # 16 train
    DYNAMIC,    # 17 motorcycle
    DYNAMIC,    # 18 bicycle
], dtype=np.int64)

ALL_SEQUENCES = list(range(22))  # KITTI odometry sequences 00..21


def labels_to_groups(label_map: np.ndarray) -> np.ndarray:
    """Map an (H, W) array of Cityscapes class ids to coarse planning group ids."""
    return CLASS_TO_GROUP[label_map]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def load_oneformer(device: str, shortest_edge: int = SHORTEST_EDGE):
    """Load the OneFormer processor + model onto `device`, in eval mode.

    Overrides the processor's resize target so inputs stay near KITTI's native
    resolution instead of being upsampled to the Cityscapes training size.
    """
    processor = OneFormerProcessor.from_pretrained(MODEL_ID)
    # Cap the resize; keep a generous longest_edge so wide KITTI frames aren't
    # squished. The image processor pads to a multiple internally.
    processor.image_processor.size = {"shortest_edge": shortest_edge,
                                      "longest_edge": shortest_edge * 4}
    model = OneFormerForUniversalSegmentation.from_pretrained(MODEL_ID).to(device).eval()
    return processor, model


def segment_batch(images: list[Image.Image], processor, model, device: str) -> list[np.ndarray]:
    """
    Run a batch of PIL images through OneFormer and return a list of (H, W)
    uint8 Cityscapes label maps (one per image). Uses fp16 autocast on CUDA.
    """
    inputs = processor(
        images=images, task_inputs=["semantic"] * len(images), return_tensors="pt"
    ).to(device)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)
    with torch.no_grad(), autocast:
        outputs = model(**inputs)
    target_sizes = [im.size[::-1] for im in images]  # (H, W) each
    segs = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
    return [s.cpu().numpy().astype(np.uint8) for s in segs]


def segment_batch_safe(images, processor, model, device, batch_size):
    """
    segment_batch with automatic back-off: on CUDA OOM, halve the batch and
    retry (recursively) so a too-large batch degrades gracefully instead of
    crashing. Returns label maps in the same order as `images`.
    """
    try:
        results = []
        for i in range(0, len(images), batch_size):
            results.extend(segment_batch(images[i:i + batch_size], processor, model, device))
        return results
    except torch.cuda.OutOfMemoryError:
        if batch_size == 1:
            raise  # a single frame won't fit -> lower SHORTEST_EDGE, don't retry forever
        gc.collect()
        torch.cuda.empty_cache()
        new_bs = max(1, batch_size // 2)
        print(f"  [OOM] backing off batch size {batch_size} -> {new_bs}")
        return segment_batch_safe(images, processor, model, device, new_bs)


# --------------------------------------------------------------------------- #
# Build: run the model over a sequence and save label maps
# --------------------------------------------------------------------------- #

def seq_out_dir(sequence_nr: int, root: Path = DATASET_ROOT) -> Path:
    """Where the label PNGs for a given sequence live."""
    return root / f"{sequence_nr:02d}"


def frame_to_pil(seq: KITTISequence, frame: int) -> Image.Image:
    img = (seq.load_image(frame) * 255).astype(np.uint8)  # (H, W, 3) in [0,1] -> uint8
    return Image.fromarray(img)


def build_dataset(sequence_nr: int, device: str | None = None, root: Path = DATASET_ROOT,
                  batch_size: int = 2, overwrite: bool = False,
                  processor=None, model=None) -> Path:
    """
    Run OneFormer over every not-yet-done frame of a KITTI sequence and save the
    raw 19-class Cityscapes label map as a single-channel uint8 PNG per frame
    (pixel value = class id 0..18). Resumable: existing PNGs are skipped and each
    result is written immediately. Returns the output directory.

    Pass a shared `processor`/`model` to avoid reloading weights per sequence.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seq = KITTISequence(sequence_nr)
    out_dir = seq_out_dir(sequence_nr, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if processor is None or model is None:
        print(f"Loading {MODEL_ID} on {device} ...")
        processor, model = load_oneformer(device)

    def out_path(frame: int) -> Path:
        return out_dir / f"{frame:06d}.png"

    todo = [f for f in range(len(seq)) if overwrite or not out_path(f).exists()]
    done_already = len(seq) - len(todo)
    if not todo:
        print(f"seq {sequence_nr:02d}: all {len(seq)} frames already done.")
        return out_dir
    print(f"seq {sequence_nr:02d}: {len(todo)} frames to do "
          f"({done_already} already present), batch size {batch_size}")

    # Prefetch/decode the next batch of images on a background thread while the
    # GPU works on the current one (image loading is pure-CPU and would
    # otherwise stall the pipeline).
    def load_chunk(frames):
        return [frame_to_pil(seq, f) for f in frames]

    chunks = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(load_chunk, chunks[0])
        for ci, frames in enumerate(tqdm(chunks, desc=f"seq {sequence_nr:02d}", unit="batch")):
            images = pending.result()
            if ci + 1 < len(chunks):                       # kick off next load now
                pending = pool.submit(load_chunk, chunks[ci + 1])
            label_maps = segment_batch_safe(images, processor, model, device, batch_size)
            for frame, label_map in zip(frames, label_maps):
                Image.fromarray(label_map).save(out_path(frame))

    return out_dir


def build_all(device: str | None = None, root: Path = DATASET_ROOT,
              batch_size: int = 2, sequences=ALL_SEQUENCES, overwrite: bool = False) -> None:
    """
    Build every sequence in `sequences`, loading the model once and reusing it.
    Prints a per-sequence timing/ETA summary and an overall total. Safe to stop
    and rerun -- finished frames are skipped.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_ID} on {device} ...")
    processor, model = load_oneformer(device)

    overall_t0 = time.perf_counter()
    for k, seq_nr in enumerate(sequences):
        t0 = time.perf_counter()
        build_dataset(seq_nr, device=device, root=root, batch_size=batch_size,
                      overwrite=overwrite, processor=processor, model=model)
        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - overall_t0
        remaining = len(sequences) - (k + 1)
        avg_per_seq = elapsed / (k + 1)
        print(f"seq {seq_nr:02d} took {dt / 60:.1f} min | "
              f"elapsed {elapsed / 60:.1f} min | "
              f"~{avg_per_seq * remaining / 60:.1f} min left over {remaining} seqs\n")

    print(f"All done in {(time.perf_counter() - overall_t0) / 60:.1f} min. Root: {root.resolve()}")


def load_label_map(sequence_nr: int, frame: int, root: Path = DATASET_ROOT) -> np.ndarray:
    """Load a previously-saved (H, W) Cityscapes class-id map for one frame."""
    path = seq_out_dir(sequence_nr, root) / f"{frame:06d}.png"
    if not path.exists():
        raise FileNotFoundError(f"No saved segmentation at {path}. Run `build` first.")
    return np.asarray(Image.open(path)).astype(np.int64)


# --------------------------------------------------------------------------- #
# Viz: show one frame's fine segmentation and coarse grouping
# --------------------------------------------------------------------------- #

def _legend(names, palette, present):
    return [Patch(color=palette[i] / 255.0, label=names[i]) for i in present]


def visualize_frame(sequence_nr: int, frame: int, root: Path = DATASET_ROOT) -> None:
    """
    Show a saved frame segmentation three ways: the KITTI image, the raw 19
    Cityscapes classes, and the coarse planning grouping (computed on the fly).
    """
    seq = KITTISequence(sequence_nr)
    image = frame_to_pil(seq, frame)
    label_map = load_label_map(sequence_nr, frame, root)
    group_map = labels_to_groups(label_map)

    fine_color = PALETTE[label_map]
    group_color = GROUP_PALETTE[group_map]

    _, axes = plt.subplots(3, 1, figsize=(11, 12))

    axes[0].imshow(image)
    axes[0].set_title(f"KITTI seq {sequence_nr:02d} frame {frame}")

    axes[1].imshow(image)
    axes[1].imshow(fine_color, alpha=0.6)
    axes[1].set_title("OneFormer — 19 Cityscapes classes")
    axes[1].legend(handles=_legend(CLASS_NAMES, PALETTE, sorted(np.unique(label_map))),
                   loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)

    axes[2].imshow(image)
    axes[2].imshow(group_color, alpha=0.6)
    axes[2].set_title("Planning-relevant groups")
    axes[2].legend(handles=_legend(GROUP_NAMES, GROUP_PALETTE, sorted(np.unique(group_map))),
                   loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    b = sub.add_parser("build", help="segment a whole sequence and save label maps")
    b.add_argument("--seq", type=int, required=True)
    b.add_argument("--device", default=None, help="cuda / cpu (auto if unset)")
    b.add_argument("--batch-size", type=int, default=1)
    b.add_argument("--overwrite", action="store_true")

    a = sub.add_parser("build-all", help="segment all sequences 00..21, resumable")
    a.add_argument("--device", default=None, help="cuda / cpu (auto if unset)")
    a.add_argument("--batch-size", type=int, default=2)
    a.add_argument("--overwrite", action="store_true")

    v = sub.add_parser("viz", help="visualise one frame's saved segmentation")
    v.add_argument("--seq", type=int, required=True)
    v.add_argument("--frame", type=int, default=0)

    args = parser.parse_args()

    if args.mode == "build":
        build_dataset(args.seq, device=args.device,
                      batch_size=args.batch_size, overwrite=args.overwrite)
    elif args.mode == "build-all":
        build_all(device=args.device, batch_size=args.batch_size, overwrite=args.overwrite)
    elif args.mode == "viz":
        visualize_frame(args.seq, args.frame)


if __name__ == "__main__":
    main()
