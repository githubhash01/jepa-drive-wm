#!/usr/bin/env python3
"""
Cache V-JEPA 2.1 ViT-G video representations for KITTI.

For consecutive KITTI frames sampled at dt = 0.5 s:

    V_t = E_video(I_t, I_{t+0.5})

where E_video is the frozen V-JEPA 2.1 ViT-G encoder
using the video tokenizer (tubelet_size = 2).

For 384 x 1248 input frames:
    H_p = 384 / 16 = 24
    W_p = 1248 / 16 = 78

so each frame pair produces:

    V_t.shape = [24, 78, 1664]

The latents land next to the other per-sequence caches:

    .../sequences/00/
        image_2/            000000.png ...
        vjepa_vitG_video/                   <- created here
            000000.npy      (24, 78, 1664)  V_0 = E(I_0, I_5)
            000005.npy      (24, 78, 1664)  V_5 = E(I_5, I_10)
            ...
            _metadata.json

File NNNNNN.npy encodes frames (NNNNNN, NNNNNN + stride). KITTI runs at
10 Hz, so the default stride of 5 frames is dt = 0.5 s, and pairs tile the
sequence back-to-back as a 2 Hz latent stream. Existing files are skipped,
so a killed run just resumes where it left off.

Needs a GPU that can hold ViT-G — run this on the workstation, not the laptop:

    PYTHONPATH=src python -m jepa_drive_wm.utils.vjepa21_video_embedder
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from jepa_drive_wm.data.kitti import KITTISequence
from jepa_drive_wm.utils.vjepa_wrapper import VJEPA21Size, VJEPA21Wrapper


class ViTG_Latent_Video_Builder:
    """Builds the vjepa_vitG_video cache for KITTI sequences.

    stride      frame gap inside a pair (5 frames = 0.5 s at 10 Hz); also the
                step between pair start frames, so pairs tile back-to-back.
    save_dtype  on-disk dtype of the .npy files (fp16 halves the ~270 GB a
                dense fp32 cache would take).
    batch_size  frame pairs per encoder forward pass; drop to 1 on OOM.
    overwrite   re-encode pairs whose .npy already exists.
    """

    OUT_DIRNAME = "vjepa_vitG_video"

    def __init__(
        self,
        stride: int = 5,
        save_dtype: np.dtype = np.float16,
        batch_size: int = 2,
        overwrite: bool = False,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = stride
        self.save_dtype = np.dtype(save_dtype)
        self.batch_size = batch_size
        self.overwrite = overwrite

        # The 5B-parameter encoder is loaded lazily so that just constructing
        # the builder (or asking it questions) stays cheap.
        self._wrapper: VJEPA21Wrapper | None = None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    @property
    def wrapper(self) -> VJEPA21Wrapper:
        if self._wrapper is None:
            self._wrapper = VJEPA21Wrapper(
                size=VJEPA21Size.GIGANTIC,   # ViT-G teacher, 1664-d
                num_frames=2,
            )
            self._wrapper.free_checkpoint_cache()
        return self._wrapper

    @property
    def layout(self):
        """Token geometry of one encoded pair: grid_t=1, (24, 78) at 384x1248."""
        return self.wrapper.layout(num_frames=2)

    def encode_pairs(self, pairs: list[tuple[str, str]]) -> torch.Tensor:
        """Encode (I_t, I_{t+stride}) image-path pairs as joint 2-frame clips.

        Returns (B, grid_h, grid_w, embed_dim) on cpu — with tubelet_size = 2
        each pair collapses into a single temporal token slice.
        """
        clips = []
        for p0, p1 in pairs:
            frames = torch.stack(
                [self.wrapper.prepare_frame(p0), self.wrapper.prepare_frame(p1)], dim=0
            )  # (2, C, H, W)
            clips.append(frames.permute(1, 0, 2, 3))  # (C, 2, H, W)
        x = torch.stack(clips, dim=0)  # (B, C, 2, H, W)

        tokens = self.wrapper._run_encoder(x)  # (B, grid_h * grid_w, D)
        lay = self.layout
        return tokens.reshape(len(pairs), lay.grid_h, lay.grid_w, lay.embed_dim)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def pair_starts(self, seq: KITTISequence) -> list[int]:
        """Start frames of the pairs to encode: every `stride` frames, keeping
        only starts whose partner frame still exists."""
        return list(range(0, len(seq) - self.stride, self.stride))

    def build_sequence(self, sequence_nr: int) -> None:
        """Encode and cache every frame pair of one KITTI sequence."""
        seq = KITTISequence(sequence_nr)
        out_dir = seq.sequence_folder / self.OUT_DIRNAME
        out_dir.mkdir(exist_ok=True)
        self.write_metadata(out_dir)

        todo = [
            i for i in self.pair_starts(seq)
            if self.overwrite or not (out_dir / f"{i:06d}.npy").exists()
        ]
        gb = len(todo) * self.bytes_per_latent / 1e9
        print(f"sequence {sequence_nr:02d}: {len(todo)} pairs to encode (~{gb:.1f} GB) -> {out_dir}")
        if not todo:
            return

        start = time.perf_counter()
        for b in range(0, len(todo), self.batch_size):
            batch = todo[b : b + self.batch_size]
            pairs = [(seq.left_images[i], seq.left_images[i + self.stride]) for i in batch]
            latents = self.encode_pairs(pairs)
            for i, latent in zip(batch, latents):
                np.save(out_dir / f"{i:06d}.npy", latent.numpy().astype(self.save_dtype, copy=False))

            done = b + len(batch)
            if done % 100 < self.batch_size or done == len(todo):
                per_pair = (time.perf_counter() - start) / done
                print(f"  {done}/{len(todo)} pairs, {per_pair:.2f} s/pair")

        print(f"sequence {sequence_nr:02d} done in {time.perf_counter() - start:.0f} s")

    def build_all(self, sequences=range(22)) -> None:
        for sequence_nr in sequences:
            self.build_sequence(sequence_nr)

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    @property
    def bytes_per_latent(self) -> int:
        lay = self.layout
        return lay.grid_h * lay.grid_w * lay.embed_dim * self.save_dtype.itemsize

    def write_metadata(self, out_dir: Path) -> None:
        """Provenance so downstream code never has to guess what's in the cache."""
        lay = self.layout
        meta = self.wrapper.metadata()
        meta["layout"] = vars(lay)
        meta["save_dtype"] = self.save_dtype.name
        meta["stride_frames"] = self.stride
        meta["array_shape"] = [lay.grid_h, lay.grid_w, lay.embed_dim]
        meta["pair_convention"] = "file NNNNNN.npy encodes frames (NNNNNN, NNNNNN + stride_frames)"
        (out_dir / "_metadata.json").write_text(json.dumps(meta, indent=2))


def build_all_video_latents(stride: int = 5, save_dtype: np.dtype = np.float16) -> None:
    """Build the ViT-G video latent cache for all 22 KITTI odometry sequences."""
    builder = ViTG_Latent_Video_Builder(stride=stride, save_dtype=save_dtype)
    builder.build_all()


if __name__ == "__main__":
    build_all_video_latents()
