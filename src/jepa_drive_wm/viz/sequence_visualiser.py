"""
Whole-sequence visualiser for KITTI: replay front-view (FV) video and watch a
BEV map get built up, live, frame by frame.

Views (all driven by `SequenceVisualiser`):

  FV RGB          -- left-camera video replay.
  FV Coarse       -- the same, recoloured by the coarse planning taxonomy.

  BEV RGB  Map    -- SLAM-style RGB map growing in the world frame.
  BEV Coarse Map  -- the same, coloured by coarse planning group.
  BEV *   Ego Map -- the accumulated map re-centred on the car (car fixed,
                     forward up; the map slides/rotates underneath it).

Each view is a matplotlib animation: `show=True` opens a window, `save=<path>`
writes a video (.mp4 via ffmpeg, .gif via pillow). The BEV backend lives in
`bev_map.BEVMapper` (lift + splat + world accumulate).

Example (high level):

    seq = KITTISequence(sequence_nr=10)
    vis = SequenceVisualiser(seq)

    vis.replay_fv("rgb")                    # FV video
    vis.replay_fv("coarse")                 # planning-relevant FV video
    vis.build_bev("rgb",    frame="world")  # RGB SLAM map building live
    vis.build_bev("coarse", frame="ego")    # coarse ego map building live

CLI:

    python -m jepa_drive_wm.viz.sequence_visualiser bev-coarse --seq 10 --stride 2
    python -m jepa_drive_wm.viz.sequence_visualiser fv-rgb --seq 10 --save /tmp/seq10.mp4
"""

import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from jepa_drive_wm.data.kitti import KITTISequence
from jepa_drive_wm.probes.dense.taxonomy import CLASS_TO_GROUP, GROUP_PALETTE
from jepa_drive_wm.viz.bev_map import BEVMapper


class SequenceVisualiser:
    def __init__(self, sequence: KITTISequence, *, resolution: float = 1.0,
                 max_depth: float = 40.0, y_range: tuple[float, float] = (-3.0, 3.0),
                 margin: float = 40.0, fps: int = 10, mask_dynamic: bool = True) -> None:
        self.seq = sequence
        self.fps = fps
        self.mapper = BEVMapper(sequence, resolution=resolution, max_depth=max_depth,
                                y_range=y_range, margin=margin, mask_dynamic=mask_dynamic)

    # ------------------------------------------------------------------ #
    # front view
    # ------------------------------------------------------------------ #

    def _fv_image(self, i: int, mode: str) -> np.ndarray:
        if mode == "rgb":
            return self.seq.get_image(i)
        if mode == "coarse":
            return GROUP_PALETTE[CLASS_TO_GROUP[self.seq.get_semantics(i)]]
        if mode == "depth":
            depth = self.seq.get_depth(i)
            depth = np.clip(depth, 0.0, 50.0) / 50.0
            return (depth * 255).astype(np.uint8)
        raise ValueError(f"mode must be 'rgb' or 'coarse', got {mode!r}")

    def replay_fv(self, mode: str = "rgb", *, start: int = 0, end: int | None = None,
                  stride: int = 1, show: bool = True, save: str | None = None):
        """Replay the front view as a video. mode: 'rgb' or 'coarse'."""
        frames = self._frames(start, end, stride)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("off")
        im = ax.imshow(self._fv_image(frames[0], mode))
        title = ax.set_title("")

        def update(k):
            fi = frames[k]
            im.set_data(self._fv_image(fi, mode))
            title.set_text(f"seq {self.seq.sequence_nr:02d}  frame {fi}  |  FV {mode}")
            return im, title

        return self._run(fig, update, len(frames), show, save)

    # ------------------------------------------------------------------ #
    # bird's-eye view map building
    # ------------------------------------------------------------------ #

    def build_bev(self, mode: str = "rgb", *, frame: str = "world", start: int = 0,
                  end: int | None = None, stride: int = 1, show: bool = True,
                  save: str | None = None, ego_fwd: float = 30.0, ego_back: float = 10.0,
                  ego_side: float = 20.0):
        """
        Build a BEV map live. mode: 'rgb' or 'coarse'. frame: 'world' (global
        SLAM-style map) or 'ego' (crop re-centred on the car each frame).
        """
        if frame not in ("world", "ego"):
            raise ValueError(f"frame must be 'world' or 'ego', got {frame!r}")
        frames = self._frames(start, end, stride)
        self.mapper.reset(mode)

        def render(fi: int) -> np.ndarray:
            if frame == "ego":
                return self.mapper.ego_image(fi, fwd=ego_fwd, back=ego_back, side=ego_side)
            img = self.mapper.world_image()                 # global map + trajectory trail
            rows, cols = self.mapper.pose_pixels(upto=fi + 1)
            for r, c in zip(rows, cols):
                if 0 <= r < self.mapper.map_h and 0 <= c < self.mapper.map_w:
                    cv2.circle(img, (c, r), 1, (255, 255, 0), -1)
            return img

        fig, ax = plt.subplots(figsize=(9, 9) if frame == "ego" else (14, 6))
        ax.axis("off")
        im = ax.imshow(render(frames[0]))                   # empty map before any add
        title = ax.set_title("")

        def update(k):
            fi = frames[k]
            self.mapper.add_frame(fi)
            im.set_data(render(fi))
            title.set_text(f"seq {self.seq.sequence_nr:02d}  frame {fi}  |  "
                           f"BEV {mode} ({frame})  |  "
                           f"{int((self.mapper.counts > 0).sum()):,} cells")
            return im, title

        return self._run(fig, update, len(frames), show, save)

    # ------------------------------------------------------------------ #
    # shared animation plumbing
    # ------------------------------------------------------------------ #

    def _frames(self, start: int, end: int | None, stride: int) -> list[int]:
        end = len(self.seq) if end is None else end
        return list(range(start, min(end, len(self.seq)), stride))

    def _run(self, fig, update, n: int, show: bool, save: str | None):
        # init_func avoids FuncAnimation calling frame 0's update twice (which
        # would double-count the first frame into the BEV accumulator).
        ani = animation.FuncAnimation(
            fig, update, init_func=lambda: [], frames=n,
            interval=1000 / self.fps, blit=False, repeat=False,
        )
        if save:
            writer = "pillow" if str(save).endswith(".gif") else "ffmpeg"
            ani.save(str(save), writer=writer, fps=self.fps)
            plt.close(fig)
            print(f"saved {save}")
        elif show:
            plt.show()
        return ani


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_VIEWS = {
    "fv-rgb":         ("replay_fv", {"mode": "rgb"}),
    "fv-coarse":      ("replay_fv", {"mode": "coarse"}),
    "fv-depth":       ("replay_fv", {"mode": "depth"}),
    "bev-rgb":        ("build_bev", {"mode": "rgb", "frame": "world"}),
    "bev-coarse":     ("build_bev", {"mode": "coarse", "frame": "world"}),
    "bev-rgb-ego":    ("build_bev", {"mode": "rgb", "frame": "ego"}),
    "bev-coarse-ego": ("build_bev", {"mode": "coarse", "frame": "ego"}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("view", choices=list(_VIEWS))
    parser.add_argument("--seq", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--keep-dynamic", action="store_true",
                        help="bake moving agents into the map (they smear); off by default")
    parser.add_argument("--save", default=None, help="write .mp4/.gif instead of showing")
    args = parser.parse_args()

    seq = KITTISequence(args.seq)
    vis = SequenceVisualiser(seq, resolution=args.resolution, fps=args.fps,
                             mask_dynamic=not args.keep_dynamic)
    method_name, kwargs = _VIEWS[args.view]
    getattr(vis, method_name)(
        **kwargs, start=args.start, end=args.end, stride=args.stride,
        show=args.save is None, save=args.save,
    )


if __name__ == "__main__":
    main()
