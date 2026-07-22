"""
Evaluate the trained world model on the held-out KITTI test sequences.

- load the best checkpoint
- metrics: per-step rollout latent L1 over the whole test set, against the
  copy-the-last-context-frame baseline (the model's own initialisation prior,
  so beating it is exactly "the model learned dynamics")
- for a few windows, visualise a 4-step autoregressive rollout by decoding the
  predicted latents with the frozen dense decoders. One depth figure and one
  semantics figure per window; each shows the pseudolabel ground truth, the
  decoder on the true latent (the decoder's ceiling) and the decoder on the
  predicted latent, so world-model error is isolated from decoder error.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from jepa_drive_wm.dense_decoder.model_depth import DepthDecoder
from jepa_drive_wm.dense_decoder.model_semantics import SemanticDecoder
from jepa_drive_wm.dense_decoder.train_depth import (
    CHECKPOINT_PATH as DEPTH_CHECKPOINT,
    MAX_DEPTH,
    MIN_DEPTH,
    predict as predict_depth,
)
from jepa_drive_wm.dense_decoder.train_semantics import (
    CHECKPOINT_PATH as SEMANTICS_CHECKPOINT,
    predict as predict_semantics,
)
from jepa_drive_wm.paths import OUTPUTS_DIR
from jepa_drive_wm.training_utils import autocast
from jepa_drive_wm.viz.visualiser import class_colors
from jepa_drive_wm.world_model.data_interface_wm import KITTIRolloutDataset
from jepa_drive_wm.world_model.model import VJEPA21WorldModel
from jepa_drive_wm.world_model.train_wm import MAX_SPEED_MPS, action_scale, latent_loss, to_grid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SEQUENCES = [4, 9]
CONTEXT_LENGTH = 4     # matches training
FRAME_STRIDE = 5       # matches training: 0.5s per step
ROLLOUT_STEPS = 4      # 1 direct prediction + 3 autoregressive

CHECKPOINT_PATH = OUTPUTS_DIR / "checkpoints_wm" / "world_model.pt"
FIGURES_DIR = OUTPUTS_DIR / "evals_wm"


@torch.no_grad()
def evaluate_rollout(model: VJEPA21WorldModel, loader, act_scale) -> tuple[list[float], list[float]]:
    """Mean latent L1 per rollout step, and the copy-last-context-frame baseline."""
    model.eval()
    model_totals = [0.0] * ROLLOUT_STEPS
    copy_totals = [0.0] * ROLLOUT_STEPS
    n_windows = 0

    for batch in loader:
        context = to_grid(batch["context_latents"].to(DEVICE), model.grid_height, model.grid_width)
        future = to_grid(batch["future_latents"].to(DEVICE), model.grid_height, model.grid_width)
        ego_motions = batch["future_ego_motions"].to(DEVICE) / act_scale

        with autocast(DEVICE):
            preds = model.rollout(context, ego_motions)  # [1, K, H, W, C]

        for k in range(ROLLOUT_STEPS):
            model_totals[k] += latent_loss(preds[:, k], future[:, k]).item()
            copy_totals[k] += latent_loss(context[:, -1], future[:, k]).item()
        n_windows += 1

    n = max(n_windows, 1)
    return [t / n for t in model_totals], [t / n for t in copy_totals]


def _save_grid(rows, step_seconds: float, title: str, path: Path) -> None:
    """One figure: rows = (label, [image per step], imshow kwargs), columns = rollout steps."""
    steps = len(rows[0][1])
    fig, axes = plt.subplots(len(rows), steps, figsize=(6 * steps, 2.1 * len(rows)))
    for row, (label, images, kwargs) in enumerate(rows):
        for k in range(steps):
            ax = axes[row, k]
            ax.imshow(images[k], **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"+{(k + 1) * step_seconds:.1f}s")
            if k == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def save_figures(
    model: VJEPA21WorldModel,
    dataset: KITTIRolloutDataset,
    depth_decoder: DepthDecoder,
    semantic_decoder: SemanticDecoder,
    act_scale,
    n_rollouts: int,
) -> None:
    """For n_rollouts evenly-spaced test windows: one depth figure and one semantics figure."""
    H, W = model.grid_height, model.grid_width

    for item in np.linspace(0, len(dataset) - 1, n_rollouts).astype(int):
        sample = dataset[item]
        sequence_nr = sample["sequence_nr"].item()
        start_index = sample["start_index"].item()
        sequence = dataset.sequences[sequence_nr]

        context = to_grid(sample["context_latents"][None].to(DEVICE), H, W)
        ego_motions = sample["future_ego_motions"][None].to(DEVICE) / act_scale
        with autocast(DEVICE):
            preds = model.rollout(context, ego_motions)[0].float()  # [K, H, W, C]

        cameras, depth_gt, depth_true, depth_pred = [], [], [], []
        sem_gt, sem_true, sem_pred = [], [], []
        for k in range(ROLLOUT_STEPS):
            frame = start_index + (CONTEXT_LENGTH + k) * FRAME_STRIDE
            gt_depth = sequence.get_depth(frame)
            gt_semantics = sequence.get_semantics(frame)
            # [1, C, H, W] for the decoders; the true latent arrives as flat tokens.
            true_chw = sample["future_latents"][k].view(H, W, -1).permute(2, 0, 1)[None].to(DEVICE)
            pred_chw = preds[k].permute(2, 0, 1)[None]

            cameras.append(sequence.get_image(frame))
            depth_gt.append(gt_depth)
            depth_true.append(predict_depth(depth_decoder, true_chw, gt_depth.shape).cpu())
            depth_pred.append(predict_depth(depth_decoder, pred_chw, gt_depth.shape).cpu())
            sem_gt.append(class_colors(gt_semantics))
            sem_true.append(class_colors(
                predict_semantics(semantic_decoder, true_chw, gt_semantics.shape).argmax(dim=1)[0].cpu().numpy()
            ))
            sem_pred.append(class_colors(
                predict_semantics(semantic_decoder, pred_chw, gt_semantics.shape).argmax(dim=1)[0].cpu().numpy()
            ))

        depth_kwargs = dict(cmap="plasma", vmin=MIN_DEPTH, vmax=MAX_DEPTH)
        stem = f"seq{sequence_nr:02d}_window{start_index:06d}"
        title = f"seq {sequence_nr:02d}, window starting at frame {start_index}"
        _save_grid(
            [
                ("camera", cameras, {}),
                ("FoundationStereo\ndepth", depth_gt, depth_kwargs),
                ("depth from\ntrue latent", depth_true, depth_kwargs),
                ("depth from\npredicted latent", depth_pred, depth_kwargs),
            ],
            dataset.step_seconds, title, FIGURES_DIR / f"{stem}_depth.png",
        )
        _save_grid(
            [
                ("camera", cameras, {}),
                ("OneFormer\nsemantics", sem_gt, {}),
                ("semantics from\ntrue latent", sem_true, {}),
                ("semantics from\npredicted latent", sem_pred, {}),
            ],
            dataset.step_seconds, title, FIGURES_DIR / f"{stem}_semantics.png",
        )


def load_decoder(cls, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    decoder = cls().to(DEVICE)
    decoder.load_state_dict(checkpoint["model_state_dict"])
    decoder.eval()
    return decoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained world model")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--rollouts", type=int, default=10,
                        help="number of evenly-spaced test windows to visualise (0 disables)")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model = VJEPA21WorldModel().to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (iter {checkpoint['iteration']}, val ar {checkpoint['val_ar']:.4f})")

    # The action normalisation is part of the model contract (see train_wm).
    assert checkpoint["max_speed_mps"] == MAX_SPEED_MPS
    act_scale = action_scale(checkpoint["step_seconds"]).to(DEVICE)

    dataset = KITTIRolloutDataset(
        TEST_SEQUENCES,
        context_length=CONTEXT_LENGTH,
        future_length=ROLLOUT_STEPS,
        frame_stride=FRAME_STRIDE,
    )
    assert dataset.step_seconds == checkpoint["step_seconds"]
    print(f"test windows: {len(dataset)} ({ROLLOUT_STEPS} steps of {dataset.step_seconds}s each)")

    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
    model_l1, copy_l1 = evaluate_rollout(model, loader, act_scale)
    for k in range(ROLLOUT_STEPS):
        print(f"step +{(k + 1) * dataset.step_seconds:.1f}s | "
              f"rollout L1 {model_l1[k]:.4f} | copy-last-frame L1 {copy_l1[k]:.4f}")

    if args.rollouts:
        depth_decoder = load_decoder(DepthDecoder, DEPTH_CHECKPOINT)
        semantic_decoder = load_decoder(SemanticDecoder, SEMANTICS_CHECKPOINT)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_figures(model, dataset, depth_decoder, semantic_decoder, act_scale, args.rollouts)
        print(f"figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
