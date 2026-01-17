"""
Policy Exporter
Runs the trained policy in Isaac Sim and saves the TRAJECTORY to a .npz file.
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path

# --- PFAD SETUP ---
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
source_dir = repo_root / "source" / "whole_body_tracking"
sys.path.insert(0, str(source_dir))

from isaaclab.app import AppLauncher
import cli_args 

parser = argparse.ArgumentParser(description="Export Policy Trajectory")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--motion_file", type=str, default=None)
parser.add_argument("--length", type=int, default=1000, help="Number of steps to record")
parser.add_argument("--filename", type=str, default="trained_motion.npz")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Headless ist OK, wir brauchen kein Bild!
args_cli.headless = True 
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    
    # 1. SETUP (WIE IM PLAY SCRIPT)
    env_cfg.scene.num_envs = 1
    
    # --- FIXES ANWENDEN ---
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.55) # Startet sicher
    if hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.resampling_time_range = (0.0, 0.0)

    # Motion File laden
    if args_cli.motion_file:
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # 2. ENVIRONMENT
    # Render mode None -> Schneller
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # 3. POLICY LADEN
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading model: {resume_path}")

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 4. RECORDING LOOP
    print(f"[INFO] Recording {args_cli.length} steps...")
    obs, _ = env.get_observations()
    
    trajectory_data = []

    for i in range(args_cli.length):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # --- EXTRACT PHYSICAL STATE ---
            # Wir holen uns die "Wahrheit" direkt aus der Simulation
            robot = env.unwrapped.scene["robot"]
            
            # Position & Rotation (World Frame)
            # Isaac: [w, x, y, z] -> Pinocchio braucht [x, y, z, w]
            root_state = robot.data.root_state_w[0].cpu().numpy()
            pos = root_state[0:3]
            quat_isaac = root_state[3:7] 
            quat_pin = np.array([quat_isaac[1], quat_isaac[2], quat_isaac[3], quat_isaac[0]])
            
            # Gelenke
            joints = robot.data.joint_pos[0].cpu().numpy()

            # Zusammenfügen für Pinocchio: [Pos(3), Rot(4), Joints(N)]
            q_frame = np.concatenate([pos, quat_pin, joints])
            trajectory_data.append(q_frame)
            
            if i % 100 == 0: print(f"Step {i}/{args_cli.length}...", end="\r")

    # 5. SPEICHERN
    out_path = os.path.join(script_dir, args_cli.filename)
    np.savez(out_path, trajectory=np.array(trajectory_data))
    print(f"\n[SUCCESS] Saved trajectory to: {out_path}")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()