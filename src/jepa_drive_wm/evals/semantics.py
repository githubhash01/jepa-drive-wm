"""
Evaluate the trained semantic decoder on the held-out KITTI test sequences.

- load the best checkpoint
- report test metrics (cross entropy / pixel accuracy) via train_semantics.evaluate
- save side-by-side (RGB | OneFormer GT | fine prediction | coarse prediction) figures
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from jepa_drive_wm.dense_decoder.data_interface_dense import KITTISemanticDataset
from jepa_drive_wm.dense_decoder.model_semantics import SemanticDecoder
from jepa_drive_wm.dense_decoder.train_semantics import (
    CHECKPOINT_PATH,
    DEVICE,
    TEST_SEQUENCES,
    evaluate,
    predict,
)
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.viz.visualiser import class_colors, group_colors

FIGURES_DIR = OUTPUTS_DIR / "evals_semantics"


@torch.no_grad()
def save_figures(model: SemanticDecoder, dataset: KITTISemanticDataset, every: int) -> None:
    """Every `every` test frames: RGB, OneFormer GT, fine and coarse predictions in one PNG."""
    for item in range(0, len(dataset), every):
        sample = dataset[item]
        sequence_nr = sample["sequence_nr"].item()
        frame_index = sample["frame_index"].item()
        target = sample["target"]  # [H, W] class ids

        logits = predict(model, sample["features"][None].to(DEVICE), target.shape)
        pred = logits.argmax(dim=1)[0].cpu().numpy()  # (H, W) ids 0..18
        image = dataset.sequences[sequence_nr].get_image(frame_index)

        panels = [
            (image, f"seq {sequence_nr:02d} frame {frame_index}"),
            (class_colors(target.numpy()), "OneFormer semantics"),
            (class_colors(pred), "predicted (fine)"),
            (group_colors(pred), "predicted (coarse)"),
        ]
        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 12))
        for ax, (panel, title) in zip(axes, panels):
            ax.imshow(panel)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"seq{sequence_nr:02d}_frame{frame_index:06d}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained semantic decoder")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--viz-every", type=int, default=100,
                        help="save a figure every N test frames (0 disables)")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model = SemanticDecoder().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"loaded {args.checkpoint} (iter {checkpoint['iteration']}, "
          f"val loss {checkpoint['validation_metrics']['loss']:.4f})")

    dataset = KITTISemanticDataset(TEST_SEQUENCES)
    metrics = evaluate(model, DataLoader(dataset, batch_size=1))
    print("test | " + " | ".join(f"{name} {value:.4f}" for name, value in metrics.items()))

    if args.viz_every:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_figures(model, dataset, args.viz_every)
        print(f"figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
