# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__, force=True)


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor


def cem(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    rollout=1,
    cem_steps=100,
    momentum_mean=0.25,
    momentum_std=0.95,
    momentum_mean_gripper=0.15,
    momentum_std_gripper=0.15,
    samples=100,
    topk=10,
    verbose=False,
    maxnorm=0.05,
    axis={},
    objective=l1,
    close_gripper=None,
):
    """
    :param context_frame: [B=1, T=1, HW, D]
    :param goal_frame: [B=1, T=1, HW, D]
    :param world_model: f(context_frame, action) -> next_frame [B, 1, HW, D]
    :return: [B=1, rollout, 7] an action trajectory over rollout horizon

    Cross-Entropy Method
    -----------------------
    1. for rollout horizon:
    1.1. sample several actions
    1.2. compute next states using WM
    3. compute similarity of final states to goal_frames
    4. select topk samples and update mean and std using topk action trajs
    5. choose final action to be mean of distribution
    """
    context_frame = context_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    goal_frame = goal_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    context_pose = context_pose.repeat(samples, 1, 1)  # Reshape to [S, 1, 7]

    # Current estimate of the mean/std of distribution over action trajectories
    mean = torch.cat(
        [
            torch.zeros((rollout, 3), device=context_frame.device),
            torch.zeros((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    std = torch.cat(
        [
            torch.ones((rollout, 3), device=context_frame.device) * maxnorm,
            torch.ones((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    def sample_action_traj():
        """Sample several action trajectories"""
        action_traj, frame_traj, pose_traj = None, context_frame, context_pose

        for h in range(rollout):

            # -- sample new action
            action_samples = torch.randn(samples, mean.size(1), device=mean.device) * std[h] + mean[h]
            action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)
            action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-0.75, max=0.75)
            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]
            action_samples = torch.cat(
                [
                    action_samples[:, :3],
                    torch.zeros((len(action_samples), 3), device=mean.device),
                    action_samples[:, -1:],
                ],
                dim=-1,
            )[:, None]
            if close_gripper is not None and h >= close_gripper:
                action_samples[:, :, -1] = 1.0

            action_traj = (
                torch.cat([action_traj, action_samples], dim=1) if action_traj is not None else action_samples
            )

            # -- compute next state
            next_frame, next_pose = world_model(frame_traj, action_traj, pose_traj)
            frame_traj = torch.cat([frame_traj, next_frame], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

        return action_traj, frame_traj

    def select_topk_action_traj(final_state, goal_state, actions):
        """Get the topk action trajectories that bring us closest to goal"""
        sims = objective(final_state.flatten(1), goal_state.flatten(1))
        indices = sims.topk(topk, largest=False).indices
        selected_actions = actions[indices]
        return selected_actions

    for step in tqdm(range(cem_steps), disable=True):
        action_traj, frame_traj = sample_action_traj()
        selected_actions = select_topk_action_traj(
            final_state=frame_traj[:, -1], goal_state=goal_frame, actions=action_traj
        )
        mean_selected_actions = selected_actions.mean(dim=0)
        std_selected_actions = selected_actions.std(dim=0)

        # -- Update new sampling mean and std based on the top-k samples
        mean = torch.cat(
            [
                mean_selected_actions[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
                mean_selected_actions[..., -1:] * (1.0 - momentum_mean_gripper)
                + mean[..., -1:] * momentum_mean_gripper,
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                std_selected_actions[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
                std_selected_actions[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
            ],
            dim=-1,
        )

        logger.info(f"new mean: {mean.sum(dim=0)} {std.sum(dim=0)}")

    new_action = torch.cat(
        [
            mean[..., :3],
            torch.zeros((rollout, 3), device=mean.device),
            round_small_elements(mean[..., -1:], 0.25),
        ],
        dim=-1,
    )[None, :]

    return new_action

def gradient_descent(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    rollout=1,
    steps=100,
    step_size=0.01,
    maxnorm=0.05,
    objective=l1,
    a_warmstart=None,
    optimize_gripper=False,
    action_l2=0.0,
    prior_l2=0.0,
    verbose=False,
):
    """
    Gradient-based action optimization.

    This is a differentiable alternative to CEM. Instead of sampling many
    action trajectories, we directly optimize the action trajectory by
    backpropagating the latent prediction loss into the action variables.

    The optimized compact action is:

        u_h = [dx, dy, dz, dg]              shape [B, rollout, 4]

    and it is expanded into the full 7D robot action:

        a_h = [dx, dy, dz, 0, 0, 0, dg]    shape [B, rollout, 7]

    Rotation deltas are fixed to zero, matching the original CEM planner.

    Objective:

        loss =
            latent_loss
            + action_l2 * ||a_xyz||^2
            + prior_l2  * ||a_xyz - a_warmstart_xyz||^2

    For diagnostic experiments, prior_l2 is the useful one: it discourages
    the optimized action from drifting far away from the warm start.
    """

    device = context_frame.device
    dtype = context_frame.dtype

    B = context_frame.size(0)
    assert B == 1, "This simple planner currently assumes B=1, matching the CEM notebook setup."

    def normalize_warmstart(a):
        """
        Accept warmstart as list, np.ndarray, or torch.Tensor.

        Expected shapes:
            [7]
            [4]
            [rollout, 7]
            [rollout, 4]
            [1, rollout, 7]
            [1, rollout, 4]
        """
        if a is None:
            return None

        if not torch.is_tensor(a):
            a = torch.tensor(a, device=device, dtype=dtype)
        else:
            a = a.to(device=device, dtype=dtype)

        if a.ndim == 1:
            # [7] or [4] -> [1, 1, 7] or [1, 1, 4]
            a = a.view(1, 1, -1)
        elif a.ndim == 2:
            # [rollout, 7] or [rollout, 4] -> [1, rollout, 7] or [1, rollout, 4]
            a = a.unsqueeze(0)
        elif a.ndim == 3:
            pass
        else:
            raise ValueError(f"a_warmstart must have ndim 1, 2, or 3, got shape {a.shape}")

        if a.shape[0] != B:
            raise ValueError(f"a_warmstart batch size {a.shape[0]} does not match B={B}")

        if a.shape[1] != rollout:
            if a.shape[1] == 1 and rollout > 1:
                # Repeat a single warm-start action across the rollout horizon.
                a = a.repeat(1, rollout, 1)
            else:
                raise ValueError(
                    f"a_warmstart rollout length {a.shape[1]} does not match rollout={rollout}"
                )

        if a.shape[-1] not in (4, 7):
            raise ValueError(
                f"a_warmstart must have last dimension 4 or 7, got shape {a.shape}"
            )

        return a

    def compact_to_full_action(v_compact):
        """
        Convert unconstrained compact variable v into a full bounded 7D action.

        v_compact: [B, rollout, 4]
        returns:   [B, rollout, 7]
        """
        xyz = maxnorm * torch.tanh(v_compact[..., :3])

        if optimize_gripper:
            gripper = 0.75 * torch.tanh(v_compact[..., -1:])
        else:
            gripper = torch.zeros_like(v_compact[..., -1:])

        zeros_rot = torch.zeros(
            (*v_compact.shape[:-1], 3),
            device=v_compact.device,
            dtype=v_compact.dtype,
        )

        full_action = torch.cat([xyz, zeros_rot, gripper], dim=-1)
        return full_action

    # ------------------------------------------------------------------
    # Build initialization and prior action.
    # ------------------------------------------------------------------

    a_warmstart = normalize_warmstart(a_warmstart)

    if a_warmstart is None:
        u_init = torch.zeros((B, rollout, 4), device=device, dtype=dtype)
        a_prior_full = None
    else:
        if a_warmstart.shape[-1] == 7:
            # Convert full action [dx, dy, dz, droll, dpitch, dyaw, dg]
            # into compact action [dx, dy, dz, dg].
            u_init = torch.cat(
                [a_warmstart[..., :3], a_warmstart[..., -1:]],
                dim=-1,
            )

            # The planner fixes rotation deltas to zero, so the prior should
            # match the action family being optimized.
            zeros_rot = torch.zeros(
                (*u_init.shape[:-1], 3),
                device=device,
                dtype=dtype,
            )
            a_prior_full = torch.cat(
                [u_init[..., :3], zeros_rot, u_init[..., -1:]],
                dim=-1,
            )
        else:
            # Compact action [dx, dy, dz, dg].
            u_init = a_warmstart

            zeros_rot = torch.zeros(
                (*u_init.shape[:-1], 3),
                device=device,
                dtype=dtype,
            )
            a_prior_full = torch.cat(
                [u_init[..., :3], zeros_rot, u_init[..., -1:]],
                dim=-1,
            )

    # ------------------------------------------------------------------
    # Use unconstrained parameter v and map to bounded actions using tanh.
    #
    #   xyz = maxnorm * tanh(v_xyz)
    #   g   = 0.75    * tanh(v_g)
    #
    # This avoids hard clipping inside the optimizer.
    # ------------------------------------------------------------------

    eps = 1e-6

    xyz0 = torch.clamp(u_init[..., :3] / maxnorm, -1.0 + eps, 1.0 - eps)
    v_xyz0 = torch.atanh(xyz0)

    if optimize_gripper:
        g0 = torch.clamp(u_init[..., -1:] / 0.75, -1.0 + eps, 1.0 - eps)
        v_g0 = torch.atanh(g0)
    else:
        v_g0 = torch.zeros((B, rollout, 1), device=device, dtype=dtype)

    v = torch.cat([v_xyz0, v_g0], dim=-1).detach().clone()
    v.requires_grad_(True)

    optimizer = torch.optim.Adam([v], lr=step_size)

    best_loss = float("inf")
    best_action = None

    # Detach fixed inputs so gradients are only used to optimize action.
    context_frame = context_frame.detach()
    context_pose = context_pose.detach()
    goal_frame = goal_frame.detach()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        action_traj = compact_to_full_action(v)

        # --------------------------------------------------------------
        # Roll out the world model.
        #
        # frame_traj starts as [B, 1, HW, D]
        # pose_traj starts as [B, 1, 7]
        #
        # At each step h:
        #   actions_so_far = [a_0, ..., a_h]
        #   next_frame, next_pose = world_model(frame_traj, actions_so_far, pose_traj)
        # --------------------------------------------------------------

        frame_traj = context_frame
        pose_traj = context_pose

        for h in range(rollout):
            actions_so_far = action_traj[:, : h + 1]

            next_frame, next_pose = world_model(
                frame_traj,
                actions_so_far,
                pose_traj,
            )

            frame_traj = torch.cat([frame_traj, next_frame], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

        final_frame = frame_traj[:, -1]

        # --------------------------------------------------------------
        # Loss components.
        # --------------------------------------------------------------

        latent_loss_vec = objective(final_frame.flatten(1), goal_frame.flatten(1))
        latent_loss = latent_loss_vec.mean()

        mag_loss = torch.zeros((), device=device, dtype=dtype)
        prior_loss = torch.zeros((), device=device, dtype=dtype)

        if action_l2 > 0.0:
            # Pull action toward zero.
            mag_loss = (action_traj[..., :3] ** 2).mean()

        if prior_l2 > 0.0 and a_prior_full is not None:
            # Pull action toward warm-start/prior.
            #
            # Only regularize xyz, because this planner fixes rotations to zero
            # and usually does not optimize gripper.
            prior_loss = ((action_traj[..., :3] - a_prior_full[..., :3]) ** 2).mean()

        loss = latent_loss + action_l2 * mag_loss + prior_l2 * prior_loss

        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())

        if loss_value < best_loss:
            best_loss = loss_value
            best_action = action_traj.detach().clone()

        if verbose and (step % 10 == 0 or step == steps - 1):
            current_action = action_traj.detach()[0, 0]
            print(
                f"[gradient] step {step:04d} "
                f"total={loss_value:.6f} "
                f"latent={float(latent_loss.detach().cpu()):.6f} "
                f"mag={float(mag_loss.detach().cpu()):.6f} "
                f"prior={float(prior_loss.detach().cpu()):.6f} "
                f"action_xyz=({current_action[0]:+.4f}, "
                f"{current_action[1]:+.4f}, "
                f"{current_action[2]:+.4f})"
            )

    return best_action

def compute_new_pose(pose, action):
    """
    :param pose: [B, T=1, 7]
    :param action: [B, T=1, 7]
    :returns: [B, T=1, 7]
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]
    # -- compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]


def poses_to_diff(start, end):
    """
    :param start: [7]
    :param end: [7]
    """
    try:
        start = start.numpy()
        end = end.numpy()
    except Exception:
        pass

    # --

    s_xyz = start[:3]
    e_xyz = end[:3]
    xyz_diff = e_xyz - s_xyz

    # --

    s_thetas = start[3:6]
    e_thetas = end[3:6]
    s_rotation = Rotation.from_euler("xyz", s_thetas, degrees=False).as_matrix()
    e_rotation = Rotation.from_euler("xyz", e_thetas, degrees=False).as_matrix()
    rotation_diff = e_rotation @ s_rotation.T
    theta_diff = Rotation.from_matrix(rotation_diff).as_euler("xyz", degrees=False)

    # --

    s_gripper = start[-1:]
    e_gripper = end[-1:]
    gripper_diff = e_gripper - s_gripper

    action = np.concatenate([xyz_diff, theta_diff, gripper_diff], axis=0)
    return torch.from_numpy(action)
