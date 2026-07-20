"""Bare-bones energy landscape example for V-JEPA 2-AC."""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "notebooks"))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from utils.world_model_wrapper import WorldModel
from app.vjepa_droid.transforms import make_transforms
from utils.mpc_utils import compute_new_pose, poses_to_diff
import time

# ---- Device ----------------------------------------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
dtype = torch.float32
print(f"Using device={device}, dtype={dtype}")

# ---- Model -----------------------------------------------------------------
print("Loading model...")
encoder, predictor = torch.hub.load(
    str(REPO_DIR), "vjepa2_ac_vit_giant", source="local", trust_repo=True,
)
print("Model loaded.")

encoder.eval().to(device=device, dtype=dtype)
predictor.eval().to(device=device, dtype=dtype)

crop_size = 256
tokens_per_frame = (crop_size // encoder.patch_size) ** 2
transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(1., 1.),
    random_resize_scale=(1., 1.),
    reprob=0., auto_augment=False, motion_shift=False,
    crop_size=crop_size,
)

# ---- Trajectory ------------------------------------------------------------
play_in_reverse = False
traj = np.load(Path(__file__).resolve().parent / "franka_example_traj.npz")
np_clips = traj["observations"]
np_states = traj["states"]
if play_in_reverse:
    np_clips = np_clips[:, ::-1].copy()
    np_states = np_states[:, ::-1].copy()
np_actions = np.expand_dims(poses_to_diff(np_states[0, 0], np_states[0, 1]), axis=(0, 1))

clips = transform(np_clips[0]).unsqueeze(0).to(device=device, dtype=dtype)
states = torch.tensor(np_states).to(device=device, dtype=dtype)
gt_actions = torch.tensor(np_actions).to(device=device, dtype=dtype)

# ---- Forward helpers -------------------------------------------------------
def forward_target(c, normalize=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    return F.layer_norm(h, (h.size(-1),)) if normalize else h


def forward_actions(z, nsamples, grid_size=0.075, normalize=True, action_repeat=1):
    grid = np.linspace(-grid_size, grid_size, nsamples)
    action_samples = torch.tensor(
        [[da, db, dc, 0, 0, 0, 0] for da in grid for db in grid for dc in grid],
        dtype=z.dtype, device=z.device,
    ).unsqueeze(1)
    S = action_samples.shape[0]

    z_hat = z[:, :tokens_per_frame].repeat(S, 1, 1)
    s_hat = states[:, :1].repeat(S, 1, 1)
    a_hat = action_samples

    for _ in range(action_repeat):
        z_next = predictor(z_hat, a_hat, s_hat)[:, -tokens_per_frame:]
        if normalize:
            z_next = F.layer_norm(z_next, (z_next.size(-1),))
        s_next = compute_new_pose(s_hat[:, -1:], a_hat[:, -1:])
        z_hat = torch.cat([z_hat, z_next], dim=1)
        s_hat = torch.cat([s_hat, s_next], dim=1)
        a_hat = torch.cat([a_hat, action_samples], dim=1)

    return z_hat, a_hat


def energy(z, h):
    return torch.abs(z[:, -tokens_per_frame:] - h[:, -tokens_per_frame:]).mean(dim=[1, 2]).tolist()


# ---- Predict optimal action via gradient-based MPC -------------------------
ground_truth_action = gt_actions[0, 0].tolist()
noise = np.random.uniform(-0.02, 0.02, size=7).tolist()
warmstart = np.array([gt + n for gt, n in zip(ground_truth_action, noise)])

world_model = WorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=tokens_per_frame,
    transform=transform,
    mpc_args={
        "rollout": 1,
        "warmstart": warmstart,
        "maxnorm": 0.15,
    },
    normalize_reps=True,
    device=device,
)

start_time = time.time()
with torch.no_grad():
    h = forward_target(clips)
z_n, z_goal = h[:, :tokens_per_frame], h[:, -tokens_per_frame:]
print("Starting planning using Gradient Descent...")
pred = world_model.infer_next_action_gradient(z_n, states[:, :1], z_goal).cpu().numpy()
end_time = time.time()

gt = gt_actions[0, 0].tolist()
print(f"Planning completed in {end_time - start_time:.2f} seconds.")
print(f"Warmstart action (x,y,z) = ({warmstart[0]:.2f},{warmstart[1]:.2f},{warmstart[2]:.2f})")
print(f"Gradient Descent predicted action (x,y,z) = ({pred[0, 0]:.2f},{pred[0, 1]:.2f},{pred[0, 2]:.2f})")
print(f"Ground truth action (x,y,z) = ({gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f})")