from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Calculate error
    error = torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1)
    resets = error > threshold
    
    # --- FORCE DEBUG: Add this temporarily! ---
    # Print the error of the first environment EVERY STEP
    if env.common_step_counter < 50: # Only print for the first 50 steps to avoid infinite spam
        print(f"DEBUG: Anchor Check Active. Current Error: {error[0].item():.4f} | Threshold: {threshold}")
    # ----------------------------------------

    if torch.any(resets):
        failed_idx = torch.where(resets)[0][0]
        err_val = error[failed_idx].item()
        print(f"[RESET] ROOT POSITION (X,Y,Z) failed! Env {failed_idx} | Err: {err_val:.4f}m > Threshold: {threshold}")
    
    return resets


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # Calculate error (Z-axis only)
    error = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    resets = error > threshold

    # --- DEBUG PRINT ---
    if torch.any(resets):
        failed_idx = torch.where(resets)[0][0]
        err_val = error[failed_idx].item()
        print(f"[RESET] ROOT HEIGHT (Z) failed! Env {failed_idx} | Err: {err_val:.4f}m > Threshold: {threshold}")
    # -------------------
    
    return resets


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    # Calculate error
    error = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs()
    resets = error > threshold

    # --- DEBUG PRINT ---
    if torch.any(resets):
        failed_idx = torch.where(resets)[0][0]
        err_val = error[failed_idx].item()
        print(f"[RESET] ORIENTATION (Gravity) failed! Env {failed_idx} | Err: {err_val:.4f} > Threshold: {threshold}")
    # -------------------

    return resets


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    
    # Calculate error (3D Euclidean distance for each limb relative to root)
    # Shape: (num_envs, num_bodies)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    
    resets = torch.any(error > threshold, dim=-1)

    # --- DEBUG PRINT ---
    if torch.any(resets):
        failed_env_idx = torch.where(resets)[0][0]
        # Find which body part failed
        body_errors = error[failed_env_idx]
        failed_body_local_idx = torch.where(body_errors > threshold)[0][0]
        
        if body_names:
            body_name = body_names[failed_body_local_idx]
        else:
            body_name = f"Body_Index_{failed_body_local_idx}"

        err_val = body_errors[failed_body_local_idx].item()
        print(f"[RESET] LIMB 3D POSITION failed! Env {failed_env_idx} | Body: {body_name} | Err: {err_val:.4f}m > Threshold: {threshold}")
    # -------------------

    return resets


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    
    # Calculate error (Z-axis only)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    resets = torch.any(error > threshold, dim=-1)

    # --- DEBUG PRINT ---
    if torch.any(resets):
        failed_env_idx = torch.where(resets)[0][0]
        # Find exactly which body part (limb) triggered the reset
        body_errors = error[failed_env_idx]
        failed_body_local_idx = torch.where(body_errors > threshold)[0][0]
        
        if body_names:
            body_name = body_names[failed_body_local_idx]
        else:
            body_name = f"Body_Index_{failed_body_local_idx}"

        err_val = body_errors[failed_body_local_idx].item()
        print(f"[RESET] LIMB Z-POS (Feet/Hands) failed! Env {failed_env_idx} | Body: {body_name} | Err: {err_val:.4f}m > Threshold: {threshold}")
    # -------------------

    return resets