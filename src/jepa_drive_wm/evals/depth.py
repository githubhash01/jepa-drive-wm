"""
Evaluate the trained depth decoder on the held-out KITTI test sequences.

- load the best checkpoint
- report test metrics (SiLog / HN / AbsRel) via train_depth.evaluate
- save side-by-side (RGB | ground truth | prediction) figures for a few frames
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from jepa_drive_wm.dense_decoder.data_interface_dense import KITTIDepthDataset
from jepa_drive_wm.dense_decoder.model_depth import DepthDecoder
from jepa_drive_wm.dense_decoder.train_depth import (
    CHECKPOINT_PATH,
    DEVICE,
    MAX_DEPTH,
    MIN_DEPTH,
    TEST_SEQUENCES,
    evaluate,
    predict,
)
from jepa_drive_wm.paths import OUTPUTS_DIR

FIGURES_DIR = OUTPUTS_DIR / "evals_depth"


@torch.no_grad()
def save_figures(model: DepthDecoder, dataset: KITTIDepthDataset, every: int) -> None:
    """Every `every` test frames: RGB, pseudolabel and prediction stacked in one PNG."""
    for item in range(0, len(dataset), every):
        sample = dataset[item]
        sequence_nr = sample["sequence_nr"].item()
        frame_index = sample["frame_index"].item()
        target = sample["target"]

        pred = predict(model, sample["features"][None].to(DEVICE), target.shape).cpu()
        image = dataset.sequences[sequence_nr].get_image(frame_index)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        axes[0].imshow(image)
        axes[0].set_title(f"seq {sequence_nr:02d} frame {frame_index}")
        axes[1].imshow(target, cmap="plasma", vmin=MIN_DEPTH, vmax=MAX_DEPTH)
        axes[1].set_title("FoundationStereo depth")
        axes[2].imshow(pred, cmap="plasma", vmin=MIN_DEPTH, vmax=MAX_DEPTH)
        axes[2].set_title("predicted depth")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"seq{sequence_nr:02d}_frame{frame_index:06d}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained depth decoder")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--viz-every", type=int, default=100,
                        help="save a figure every N test frames (0 disables)")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model = DepthDecoder().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"loaded {args.checkpoint} (iter {checkpoint['iteration']}, "
          f"val total {checkpoint['validation_metrics']['total']:.4f})")

    dataset = KITTIDepthDataset(TEST_SEQUENCES)
    metrics = evaluate(model, DataLoader(dataset, batch_size=1))
    print("test | " + " | ".join(f"{name} {value:.4f}" for name, value in metrics.items()))

    if args.viz_every:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_figures(model, dataset, args.viz_every)
        print(f"figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
