"""
Run the V-JEPA 2-AC energy-landscape demo from a plain Python script.

This script is a cleaned-up version of the notebook-style experiment. It:
  1. Loads the pretrained V-JEPA 2-AC encoder and predictor from a local repo.
  2. Loads the example Franka trajectory.
  3. Encodes the trajectory frames into V-JEPA latent tokens.
  4. Sweeps a 3D grid of candidate Cartesian actions.
  5. Scores each action by latent prediction error.
  6. Saves trajectory and energy-landscape plots.
  7. Optionally runs the CEM/MPC planner from the official notebook utilities.

Typical usage from the repository root:

    cd /home/hashim/PLSLAM/vjepa2
    conda activate vjepa2_env
    python energy_landscape_clean.py \
        --repo-dir /home/hashim/PLSLAM/vjepa2 \
        --trajectory notebooks/franka_example_traj.npz \
        --device cpu \
        --output-dir outputs/energy_landscape

The model checkpoint should already exist at:

    ~/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

# Use a non-interactive backend so the script runs cleanly over SSH/terminal.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ExperimentConfig:
    repo_dir: Path
    trajectory_path: Path
    output_dir: Path
    device: str
    crop_size: int
    nsamples: int
    grid_size: float
    action_repeat: int
    grid_batch_size: int
    play_in_reverse: bool
    normalize_reps: bool
    run_cem: bool
    cem_samples: int
    cem_topk: int
    cem_steps: int
    cem_rollout: int


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="V-JEPA 2-AC action energy-landscape demo."
    )

    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("/home/hashim/PLSLAM/vjepa2"),
        help="Path to the local vjepa2 repository.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("notebooks/franka_example_traj.npz"),
        help="Path to the Franka trajectory .npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/energy_landscape"),
        help="Directory where plots and NumPy outputs will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use, e.g. 'cpu' or 'cuda:0'. CPU is safest for memory.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Input crop size used by the V-JEPA 2-AC model.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=5,
        help="Number of grid samples per action axis. Total grid size is nsamples^3.",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=0.075,
        help="Candidate action range for x/y/z: [-grid_size, grid_size].",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="How many times to roll the same candidate action forward.",
    )
    parser.add_argument(
        "--grid-batch-size",
        type=int,
        default=64,
        help="Number of candidate actions to score per predictor call.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Load the trajectory backwards to inspect how the landscape changes.",
    )
    parser.add_argument(
        "--no-normalize-reps",
        action="store_true",
        help="Disable layer normalization of encoder/predictor representations.",
    )
    parser.add_argument(
        "--no-cem",
        action="store_true",
        help="Skip the CEM planner and only compute the energy grid.",
    )
    parser.add_argument("--cem-samples", type=int, default=25)
    parser.add_argument("--cem-topk", type=int, default=10)
    parser.add_argument("--cem-steps", type=int, default=2)
    parser.add_argument("--cem-rollout", type=int, default=2)

    args = parser.parse_args()

    return ExperimentConfig(
        repo_dir=args.repo_dir.expanduser().resolve(),
        trajectory_path=args.trajectory.expanduser(),
        output_dir=args.output_dir.expanduser(),
        device=args.device,
        crop_size=args.crop_size,
        nsamples=args.nsamples,
        grid_size=args.grid_size,
        action_repeat=args.action_repeat,
        grid_batch_size=args.grid_batch_size,
        play_in_reverse=args.reverse,
        normalize_reps=not args.no_normalize_reps,
        run_cem=not args.no_cem,
        cem_samples=args.cem_samples,
        cem_topk=args.cem_topk,
        cem_steps=args.cem_steps,
        cem_rollout=args.cem_rollout,
    )


def add_repo_paths(repo_dir: Path) -> None:
    """Make repo-local packages and notebook utilities importable."""
    candidates = [repo_dir, repo_dir / "notebooks"]
    for path in candidates:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def check_checkpoint_exists() -> Path:
    """Check that the V-JEPA 2-AC checkpoint is in Torch Hub's cache."""
    ckpt_path = Path(torch.hub.get_dir()) / "checkpoints" / "vjepa2-ac-vitg.pt"
    print(f"[checkpoint] path: {ckpt_path}")
    print(f"[checkpoint] exists: {ckpt_path.exists()}")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            "Missing V-JEPA 2-AC checkpoint. Expected it at:\n"
            f"  {ckpt_path}\n"
            "Download it with:\n"
            "  wget -O ~/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt "
            "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
        )

    print(f"[checkpoint] size: {ckpt_path.stat().st_size / 1e9:.2f} GB")
    return ckpt_path


def load_model(repo_dir: Path, device: str):
    """Load the pretrained V-JEPA 2-AC encoder and action-conditioned predictor."""
    print("[model] loading pretrained V-JEPA 2-AC model")
    print(f"[model] repo: {repo_dir}")
    print(f"[model] device: {device}")

    encoder, predictor = torch.hub.load(
        str(repo_dir),
        "vjepa2_ac_vit_giant",
        source="local",
        trust_repo=True,
    )

    encoder.eval().to(device)
    predictor.eval().to(device)

    print("[model] loaded")
    print(f"[model] encoder patch_size: {encoder.patch_size}")
    return encoder, predictor


def build_transform(crop_size: int):
    """Create the deterministic transform used in the original notebook."""
    from app.vjepa_droid.transforms import make_transforms

    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )


def load_trajectory(config: ExperimentConfig, transform) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Load the example trajectory and convert it into model-ready tensors."""
    from utils.mpc_utils import poses_to_diff

    trajectory_path = config.trajectory_path
    if not trajectory_path.is_absolute():
        trajectory_path = config.repo_dir / trajectory_path

    print(f"[data] loading trajectory: {trajectory_path}")
    trajectory = np.load(trajectory_path)
    np_clips = trajectory["observations"]
    np_states = trajectory["states"]

    if config.play_in_reverse:
        print("[data] reversing trajectory")
        np_clips = np_clips[:, ::-1].copy()
        np_states = np_states[:, ::-1].copy()

    # The notebook uses the action between the first two states as ground truth.
    np_actions = np.expand_dims(
        poses_to_diff(np_states[0, 0], np_states[0, 1]),
        axis=(0, 1),
    )

    # transform(np_clips[0]) returns [C, T, H, W]. Add batch dimension.
    clips = transform(np_clips[0]).unsqueeze(0).to(config.device, non_blocking=True)
    states = torch.as_tensor(np_states, dtype=torch.float32, device=config.device)
    actions = torch.as_tensor(np_actions, dtype=torch.float32, device=config.device)

    print(f"[data] clips:  {tuple(clips.shape)}")
    print(f"[data] states: {tuple(states.shape)}")
    print(f"[data] action: {tuple(actions.shape)}")
    print(
        "[data] ground-truth action (x,y,z): "
        f"({actions[0, 0, 0].item():+.4f}, "
        f"{actions[0, 0, 1].item():+.4f}, "
        f"{actions[0, 0, 2].item():+.4f})"
    )

    return clips, states, actions, np_clips


def save_video_strip(np_clips: np.ndarray, output_dir: Path) -> Path:
    """Save a horizontal strip showing the frames in the loaded trajectory."""
    clip = np_clips[0]
    num_frames = len(clip)

    # Original notebook layout: [T, H, W, C] -> one wide image.
    strip = np.transpose(clip, (1, 0, 2, 3)).reshape(clip.shape[1], clip.shape[2] * num_frames, 3)

    fig, ax = plt.subplots(figsize=(20, 3))
    ax.imshow(strip)
    ax.set_title("Loaded trajectory frames")
    ax.axis("off")
    fig.tight_layout()

    out_path = output_dir / "trajectory_frames.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved frame strip: {out_path}")
    return out_path


def encode_targets(
    encoder,
    clips: torch.Tensor,
    normalize_reps: bool,
) -> torch.Tensor:
    """Encode each frame into V-JEPA latent tokens.

    Input shape:
        clips: [B, C, T, H, W]

    Output shape:
        h: [B, T * tokens_per_frame, D]
    """
    with torch.no_grad():
        batch_size, channels, num_frames, height, width = clips.size()

        # The V-JEPA encoder expects two-frame tubelets. The notebook duplicates each
        # single frame to make a two-frame tubelet per time index.
        encoder_input = (
            clips.permute(0, 2, 1, 3, 4)
            .flatten(0, 1)
            .unsqueeze(2)
            .repeat(1, 1, 2, 1, 1)
        )

        h = encoder(encoder_input)
        h = h.view(batch_size, num_frames, -1, h.size(-1)).flatten(1, 2)

        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))

    print(f"[encode] target reps: {tuple(h.shape)}")
    return h


def make_action_grid(
    nsamples: int,
    grid_size: float,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create Cartesian action candidates over x/y/z with remaining action dims zero.

    Returns:
        action_grid: [S, 1, 7], where S = nsamples^3.
    """
    values = np.linspace(-grid_size, grid_size, nsamples, dtype=np.float32)
    action_samples = []

    for dx in values:
        for dy in values:
            for dz in values:
                action_samples.append([dx, dy, dz, 0.0, 0.0, 0.0, 0.0])

    action_grid = torch.tensor(action_samples, device=device, dtype=dtype).unsqueeze(1)
    print(f"[grid] sampled {len(action_grid)} actions ({nsamples}^3)")
    return action_grid


def iter_chunks(x: torch.Tensor, chunk_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, len(x), chunk_size):
        yield x[start : start + chunk_size]


def score_action_grid(
    predictor,
    target_reps: torch.Tensor,
    states: torch.Tensor,
    tokens_per_frame: int,
    action_grid: torch.Tensor,
    normalize_reps: bool,
    action_repeat: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score each candidate action by predicted-vs-target latent error.

    The score is the mean absolute error between the final predicted latent frame
    and the final target latent frame from the encoded trajectory.

    Returns:
        used_actions_xyz: [S, 3], total x/y/z action actually applied.
        energies: [S], lower is better.
    """
    all_used_actions = []
    all_energies = []

    target_final = target_reps[:, -tokens_per_frame:]
    context_frame = target_reps[:, :tokens_per_frame]
    context_pose = states[:, :1]

    for chunk_idx, action_chunk in enumerate(iter_chunks(action_grid, batch_size), start=1):
        num_candidates = action_chunk.size(0)

        z_hat = context_frame.repeat(num_candidates, 1, 1)
        s_hat = context_pose.repeat(num_candidates, 1, 1)
        a_hat = action_chunk

        # Repeatedly apply the same candidate action. The predictor receives the
        # full token/action/state history accumulated so far.
        for _ in range(action_repeat):
            pred_next = predictor(z_hat, a_hat, s_hat)[:, -tokens_per_frame:]
            if normalize_reps:
                pred_next = F.layer_norm(pred_next, (pred_next.size(-1),))

            from utils.mpc_utils import compute_new_pose

            next_pose = compute_new_pose(s_hat[:, -1:], a_hat[:, -1:])

            z_hat = torch.cat([z_hat, pred_next], dim=1)
            s_hat = torch.cat([s_hat, next_pose], dim=1)
            a_hat = torch.cat([a_hat, action_chunk], dim=1)

        pred_final = z_hat[:, -tokens_per_frame:]
        energy = torch.abs(pred_final - target_final).mean(dim=(1, 2))

        # The last action in a_hat is prepared for the next step, so the actions
        # actually used in the rollout are a_hat[:, :-1].
        used_action_xyz = a_hat[:, :-1, :3].sum(dim=1)

        all_used_actions.append(used_action_xyz.detach().cpu().numpy())
        all_energies.append(energy.detach().cpu().numpy())

        print(
            f"[grid] scored chunk {chunk_idx}: "
            f"{num_candidates} candidates; min energy={energy.min().item():.6f}"
        )

    used_actions_xyz = np.concatenate(all_used_actions, axis=0)
    energies = np.concatenate(all_energies, axis=0)
    return used_actions_xyz, energies


def save_energy_plots(
    used_actions_xyz: np.ndarray,
    energies: np.ndarray,
    gt_action_xyz: np.ndarray,
    output_dir: Path,
    nsamples: int,
) -> None:
    """Save 2D energy views and a 3D scatter plot."""
    best_idx = int(np.argmin(energies))
    best_action = used_actions_xyz[best_idx]
    best_energy = float(energies[best_idx])

    print(
        "[grid] best grid action (x,y,z): "
        f"({best_action[0]:+.4f}, {best_action[1]:+.4f}, {best_action[2]:+.4f}); "
        f"energy={best_energy:.6f}"
    )
    print(
        "[grid] ground-truth action (x,y,z): "
        f"({gt_action_xyz[0]:+.4f}, {gt_action_xyz[1]:+.4f}, {gt_action_xyz[2]:+.4f})"
    )

    np.savez(
        output_dir / "energy_grid_results.npz",
        used_actions_xyz=used_actions_xyz,
        energies=energies,
        gt_action_xyz=gt_action_xyz,
        best_action_xyz=best_action,
        best_energy=best_energy,
    )

    axis_pairs = [
        (0, 2, "x", "z", "energy_xz.png"),
        (0, 1, "x", "y", "energy_xy.png"),
        (1, 2, "y", "z", "energy_yz.png"),
    ]

    for i, j, label_i, label_j, filename in axis_pairs:
        fig, ax = plt.subplots(figsize=(7, 6))

        # Since each grid point is unique, histogram2d with weighted mean is a
        # convenient way to produce the same style as the original notebook.
        heatmap, xedges, yedges = np.histogram2d(
            used_actions_xyz[:, i],
            used_actions_xyz[:, j],
            weights=energies,
            bins=nsamples,
        )
        counts, _, _ = np.histogram2d(
            used_actions_xyz[:, i],
            used_actions_xyz[:, j],
            bins=[xedges, yedges],
        )
        heatmap = np.divide(heatmap, counts, out=np.full_like(heatmap, np.nan), where=counts > 0)

        im = ax.imshow(
            heatmap.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
        )
        ax.scatter(gt_action_xyz[i], gt_action_xyz[j], marker="x", s=120, label="ground truth")
        ax.scatter(best_action[i], best_action[j], marker="o", s=90, facecolors="none", label="best grid")
        ax.set_xlabel(f"Action delta {label_i}")
        ax.set_ylabel(f"Action delta {label_j}")
        ax.set_title(f"Energy landscape ({label_i}-{label_j})")
        ax.legend(loc="best")
        fig.colorbar(im, ax=ax, label="mean latent prediction error")
        fig.tight_layout()

        out_path = output_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] saved: {out_path}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        used_actions_xyz[:, 0],
        used_actions_xyz[:, 1],
        used_actions_xyz[:, 2],
        c=energies,
        s=40,
    )
    ax.scatter(gt_action_xyz[0], gt_action_xyz[1], gt_action_xyz[2], marker="x", s=120, label="ground truth")
    ax.scatter(best_action[0], best_action[1], best_action[2], marker="o", s=90, label="best grid")
    ax.set_xlabel("Action delta x")
    ax.set_ylabel("Action delta y")
    ax.set_zlabel("Action delta z")
    ax.set_title("3D action energy grid")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="latent prediction error")
    fig.tight_layout()

    out_path = output_dir / "energy_3d.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved: {out_path}")


def run_cem_planner(
    encoder,
    predictor,
    transform,
    tokens_per_frame: int,
    target_reps: torch.Tensor,
    states: torch.Tensor,
    gt_action_xyz: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """Run the official CEM planner wrapper from the notebook utilities."""
    from utils.world_model_wrapper import WorldModel

    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args={
            "rollout": config.cem_rollout,
            "samples": config.cem_samples,
            "topk": config.cem_topk,
            "cem_steps": config.cem_steps,
            "momentum_mean": 0.15,
            "momentum_mean_gripper": 0.15,
            "momentum_std": 0.75,
            "momentum_std_gripper": 0.15,
            "maxnorm": config.grid_size,
            "verbose": True,
        },
        normalize_reps=config.normalize_reps,
        device=config.device,
    )

    z_start = target_reps[:, :tokens_per_frame]
    z_goal = target_reps[:, -tokens_per_frame:]
    start_state = states[:, :1]

    print("[cem] starting Cross-Entropy Method planning")
    with torch.no_grad():
        planned_action = world_model.infer_next_action(z_start, start_state, z_goal)

    planned_action_np = planned_action.detach().cpu().numpy()
    print(
        "[cem] planned action (x,y,z): "
        f"({planned_action_np[0, 0]:+.4f}, "
        f"{planned_action_np[0, 1]:+.4f}, "
        f"{planned_action_np[0, 2]:+.4f})"
    )
    print(
        "[cem] ground truth (x,y,z):     "
        f"({gt_action_xyz[0]:+.4f}, {gt_action_xyz[1]:+.4f}, {gt_action_xyz[2]:+.4f})"
    )
    return planned_action_np


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    add_repo_paths(config.repo_dir)
    check_checkpoint_exists()

    encoder, predictor = load_model(config.repo_dir, config.device)
    transform = build_transform(config.crop_size)

    tokens_per_frame = int((config.crop_size // encoder.patch_size) ** 2)
    print(f"[model] tokens_per_frame: {tokens_per_frame}")

    clips, states, gt_actions, np_clips = load_trajectory(config, transform)
    save_video_strip(np_clips, config.output_dir)

    target_reps = encode_targets(encoder, clips, config.normalize_reps)

    action_grid = make_action_grid(
        nsamples=config.nsamples,
        grid_size=config.grid_size,
        device=config.device,
        dtype=target_reps.dtype,
    )

    with torch.no_grad():
        used_actions_xyz, energies = score_action_grid(
            predictor=predictor,
            target_reps=target_reps,
            states=states,
            tokens_per_frame=tokens_per_frame,
            action_grid=action_grid,
            normalize_reps=config.normalize_reps,
            action_repeat=config.action_repeat,
            batch_size=config.grid_batch_size,
        )

    gt_action_xyz = gt_actions[0, 0, :3].detach().cpu().numpy()
    save_energy_plots(
        used_actions_xyz=used_actions_xyz,
        energies=energies,
        gt_action_xyz=gt_action_xyz,
        output_dir=config.output_dir,
        nsamples=config.nsamples,
    )

    if config.run_cem:
        planned_action = run_cem_planner(
            encoder=encoder,
            predictor=predictor,
            transform=transform,
            tokens_per_frame=tokens_per_frame,
            target_reps=target_reps,
            states=states,
            gt_action_xyz=gt_action_xyz,
            config=config,
        )
        np.save(config.output_dir / "cem_action.npy", planned_action)
    else:
        print("[cem] skipped")

    print(f"[done] outputs written to: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()