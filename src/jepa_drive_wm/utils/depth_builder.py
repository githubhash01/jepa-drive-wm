#!/usr/bin/env python3
"""
Faster KITTI depth-map generation using FoundationStereo.

Compared with the original helper, this version adds:
  * optional batch inference (--batch_size)
  * lower PNG compression for faster writes (--png_compression)
  * cv2 image decoding instead of imageio
  * optional full-resolution output after scaled inference (--save_full_res)
  * optional CUDA speed knobs: TF32, channels_last, torch.compile
  * more useful per-frame timing for tuning

The main quality/speed knobs are still --scale and --valid_iters.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


def setup_torch_speed_knobs():
    """Safe-ish CUDA performance settings for inference."""
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def build_model(fs_root: Path, ckpt_path: Path, *, channels_last: bool = False, compile_model: bool = False):
    """Construct FoundationStereo and load weights once."""
    sys.path.append(str(fs_root))
    from core.foundation_stereo import FoundationStereo

    cfg = OmegaConf.load(str(ckpt_path.parent / "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    model_args = OmegaConf.create(dict(cfg))

    model = FoundationStereo(model_args)
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    print(f"ckpt global_step:{ckpt.get('global_step')}, epoch:{ckpt.get('epoch')}")
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if compile_model:
        # This can give a useful speedup on long runs, but the first frame will be slow
        # because PyTorch compiles the graph. If it fails, fall back to eager mode.
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("torch.compile enabled")
        except Exception as exc:
            print(f"torch.compile failed; continuing without it: {exc}")

    return model


def read_intrinsics(k_path: Path):
    """K.txt: line 1 = flattened 3x3 K, line 2 = baseline in metres."""
    with open(k_path, "r") as f:
        lines = f.readlines()
    K = np.array(list(map(float, lines[0].split())), dtype=np.float64).reshape(3, 3)
    baseline = float(lines[1])
    return K, baseline


def read_rgb(path: Path) -> np.ndarray:
    """Fast PNG/JPEG decode; returns RGB HxWx3 uint8 like imageio.imread."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def prepare_pair(left_path: Path, right_path: Path, scale: float):
    img0 = read_rgb(left_path)
    img1 = read_rgb(right_path)
    orig_shape = img0.shape[:2]

    if img1.shape[:2] != orig_shape:
        raise ValueError(f"Left/right shapes differ: {left_path} {img0.shape}, {right_path} {img1.shape}")

    if scale != 1.0:
        img0 = cv2.resize(img0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(img0), np.ascontiguousarray(img1), orig_shape


@torch.inference_mode()
def infer_depth_batch(
    model,
    items,
    K,
    baseline: float,
    scale: float,
    valid_iters: int,
    z_far: float,
    *,
    save_full_res: bool = False,
    channels_last: bool = False,
    amp_dtype: str = "fp16",
):
    """Run one batched inference pass. items are (fid, left_path, right_path, out_path)."""
    from core.utils.utils import InputPadder

    lefts, rights, orig_shapes = [], [], []
    scaled_shape = None
    for _, left_path, right_path, _ in items:
        img0, img1, orig_shape = prepare_pair(left_path, right_path, scale)
        if scaled_shape is None:
            scaled_shape = img0.shape[:2]
        elif img0.shape[:2] != scaled_shape:
            raise ValueError(
                "All images in a batch must have the same scaled size. "
                "Use --batch_size 1 for variable-sized inputs."
            )
        lefts.append(img0)
        rights.append(img1)
        orig_shapes.append(orig_shape)

    batch = len(items)
    H, W = scaled_shape
    arr0 = np.stack(lefts, axis=0)
    arr1 = np.stack(rights, axis=0)

    t0 = torch.from_numpy(arr0).permute(0, 3, 1, 2).to("cuda", dtype=torch.float32, non_blocking=True)
    t1 = torch.from_numpy(arr1).permute(0, 3, 1, 2).to("cuda", dtype=torch.float32, non_blocking=True)
    if channels_last:
        t0 = t0.contiguous(memory_format=torch.channels_last)
        t1 = t1.contiguous(memory_format=torch.channels_last)

    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)

    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
        disp = model.forward(t0, t1, iters=valid_iters, test_mode=True)

    disp = padder.unpad(disp.float())
    if disp.ndim == 4:
        disp = disp[:, 0]
    elif disp.ndim == 2:
        disp = disp[None]
    elif disp.ndim != 3:
        disp = disp.reshape(batch, H, W)

    disp_np = disp.detach().cpu().numpy()

    fx = K[0, 0] * scale
    depths = []
    for i in range(batch):
        d = disp_np[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = fx * baseline / d
        depth[~np.isfinite(depth)] = 0.0
        depth[d <= 0] = 0.0
        depth = np.clip(depth, 0.0, z_far)

        if save_full_res and scale != 1.0:
            oh, ow = orig_shapes[i]
            # Depth values are already metric. Resize values only; do not rescale them.
            depth = cv2.resize(depth, (ow, oh), interpolation=cv2.INTER_LINEAR)

        depths.append(depth)

    return depths


def save_depth_png(depth: np.ndarray, out_path: Path, compression: int):
    """16-bit PNG, value = depth_m * 256, KITTI convention."""
    depth_u16 = np.clip(depth * 256.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(out_path), depth_u16, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", type=int, default=4)
    ap.add_argument("--kitti_root", default="/home/hashim/Desktop/Datasets/KITTI")
    ap.add_argument("--fs_root", default="/home/hashim/Modules/FoundationStereo")
    ap.add_argument(
        "--ckpt",
        default="/home/hashim/Modules/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth",
    )
    ap.add_argument("--z_far", type=float, default=80.0)
    ap.add_argument("--valid_iters", type=int, default=16)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--depth_dirname", default="depth_fast")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4, help="Try 2 or 4 if VRAM allows; use 1 if you hit OOM.")
    ap.add_argument("--png_compression", type=int, default=1, choices=range(10), metavar="0-9")
    ap.add_argument("--save_full_res", action="store_true", help="Upsample scaled depth back to the original KITTI image size.")
    ap.add_argument("--channels_last", action="store_true", help="May speed up conv-heavy parts on recent NVIDIA GPUs.")
    ap.add_argument("--compile", action="store_true", help="Try torch.compile. First frame will be much slower; useful for long runs.")
    ap.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--max_frames", type=int, default=0, help="For quick benchmarking; 0 means all frames.")
    args = ap.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    setup_torch_speed_knobs()

    seq_dir = Path(args.kitti_root) / f"data_odometry_color/dataset/sequences/{args.sequence:02d}"
    left_dir = seq_dir / "image_2"
    right_dir = seq_dir / "image_3"
    k_path = seq_dir / "K.txt"
    depth_dir = seq_dir / args.depth_dirname
    depth_dir.mkdir(parents=True, exist_ok=True)

    if not k_path.exists():
        raise FileNotFoundError(f"{k_path} not found. Run write_foundationstereo_intrinsics first.")

    frame_ids = sorted(p.stem for p in left_dir.glob("*.png"))
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise FileNotFoundError(f"No frames found in {left_dir}")

    K, baseline = read_intrinsics(k_path)
    print(f"Sequence {args.sequence:02d}: {len(frame_ids)} frames -> {depth_dir}")
    print(f"fx={K[0,0]:.3f} baseline={baseline:.4f}m scale={args.scale} iters={args.valid_iters} batch={args.batch_size}")

    model = build_model(
        Path(args.fs_root),
        Path(args.ckpt),
        channels_last=args.channels_last,
        compile_model=args.compile,
    )

    work = []
    for fid in frame_ids:
        out_path = depth_dir / f"{fid}.png"
        if out_path.exists() and not args.overwrite:
            continue
        left_path = left_dir / f"{fid}.png"
        right_path = right_dir / f"{fid}.png"
        if not right_path.exists():
            print(f"  skip {fid}: missing right image")
            continue
        work.append((fid, left_path, right_path, out_path))

    print(f"Frames to process: {len(work)}")
    t_start = time.perf_counter()
    done = 0

    for batch_items in chunks(work, args.batch_size):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        depths = infer_depth_batch(
            model,
            batch_items,
            K,
            baseline,
            args.scale,
            args.valid_iters,
            args.z_far,
            save_full_res=args.save_full_res,
            channels_last=args.channels_last,
            amp_dtype=args.amp_dtype,
        )
        for depth, (_, _, _, out_path) in zip(depths, batch_items):
            save_depth_png(depth, out_path, args.png_compression)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        done += len(batch_items)

        if done == len(batch_items) or done % 20 == 0 or done >= len(work):
            elapsed = time.perf_counter() - t_start
            print(
                f"  {batch_items[-1][0]}  batch {dt:.2f}s, {dt/len(batch_items):.2f}s/frame "
                f"({done} done, avg {elapsed/max(done,1):.2f}s/frame)"
            )

    elapsed = time.perf_counter() - t_start
    print(f"Finished {done} frames in {elapsed:.1f}s ({elapsed/max(done,1):.2f}s/frame)")


if __name__ == "__main__":
    main()
