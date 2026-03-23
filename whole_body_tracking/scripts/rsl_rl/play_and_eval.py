"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
source_dir = repo_root / "source" / "whole_body_tracking"
sys.path.insert(0, str(source_dir))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--motion_file", type=str, default=None)
parser.add_argument("--registry_name", type=str, default=None)
# NEW: how many trials to evaluate before printing stats
parser.add_argument("--eval_episodes", type=int, default=100,
                    help="Number of complete episodes to evaluate for success rate.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class BackflipSuccessTracker:
    def __init__(self, num_envs: int):
        self.num_envs    = num_envs
        self.completed   = 0
        self.max_pitch   = torch.zeros(num_envs)
        self.landed_upright = torch.zeros(num_envs, dtype=torch.bool)

        # Graduated success bins
        self.bins = {
            "full_success":    0,   # >=300° + landed upright + no fall
            "rotated_landed":  0,   # >=300° + landed upright (may have fallen)
            "rotated_only":    0,   # >=300° rotation seen
            "partial_180":     0,   # >=180° rotation
            "partial_90":      0,   # >=90° rotation
            "no_rotation":     0,   # <90°
        }

    def update(self, base_quat_w, projected_gravity):
        w = base_quat_w[:, 0]
        x = base_quat_w[:, 1]
        y = base_quat_w[:, 2]
        z = base_quat_w[:, 3]
        pitch_deg = torch.abs(
            torch.arcsin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        ) * 180.0 / torch.pi
        self.max_pitch = torch.maximum(self.max_pitch, pitch_deg.cpu())
        upright = projected_gravity[:, 2].cpu() < -0.7
        self.landed_upright = self.landed_upright | upright

    def register_episode_end(self, dones, timed_out):
        finished = dones.cpu().bool()
        for i in range(self.num_envs):
            if finished[i]:
                self.completed += 1
                p          = self.max_pitch[i].item()
                landed     = self.landed_upright[i].item()
                not_fallen = timed_out[i].item() if timed_out is not None else True

                if p >= 300 and landed and not_fallen:
                    self.bins["full_success"] += 1
                elif p >= 300 and landed:
                    self.bins["rotated_landed"] += 1
                elif p >= 300:
                    self.bins["rotated_only"] += 1
                elif p >= 180:
                    self.bins["partial_180"] += 1
                elif p >= 90:
                    self.bins["partial_90"] += 1
                else:
                    self.bins["no_rotation"] += 1

                if self.completed <= 10:
                    print(f"  [Ep {self.completed}] env={i}  "
                          f"pitch={p:.1f}°  upright={landed}  "
                          f"timed_out={not_fallen}")

                self.max_pitch[i]      = 0.0
                self.landed_upright[i] = False

    def summarize(self):
        n = max(self.completed, 1)
        print("\n" + "="*55)
        print(f"  BACKFLIP EVALUATION  ({self.completed} episodes)")
        print("="*55)
        print(f"  Full success (≥300°, landed, no fall): "
              f"{self.bins['full_success']:3d}  ({self.bins['full_success']/n*100:.1f}%)")
        print(f"  Rotated + landed (≥300°, fell):        "
              f"{self.bins['rotated_landed']:3d}  ({self.bins['rotated_landed']/n*100:.1f}%)")
        print(f"  Rotated only (≥300°, no landing):      "
              f"{self.bins['rotated_only']:3d}  ({self.bins['rotated_only']/n*100:.1f}%)")
        print(f"  Partial 180°–300°:                     "
              f"{self.bins['partial_180']:3d}  ({self.bins['partial_180']/n*100:.1f}%)")
        print(f"  Partial 90°–180°:                      "
              f"{self.bins['partial_90']:3d}  ({self.bins['partial_90']/n*100:.1f}%)")
        print(f"  No rotation (<90°):                    "
              f"{self.bins['no_rotation']:3d}  ({self.bins['no_rotation']/n*100:.1f}%)")
        print("="*55)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )

    if args_cli.motion_file:
        print(f"[INFO] Using LOCAL motion file: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    elif args_cli.wandb_path:
        import wandb
        run_path = args_cli.wandb_path
        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [f.name for f in wandb_run.files() if "model" in f.name]
        file = (args_cli.wandb_path.split("/")[-1] if "model" in args_cli.wandb_path
                else max(files, key=lambda x: int(x.split("_")[1].split(".")[0])))
        wandb_run.file(str(file)).download("./logs/rsl_rl/temp", replace=True)
        resume_path = f"./logs/rsl_rl/temp/{file}"
        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art:
            env_cfg.commands.motion.motion_file = str(Path(art.download()) / "motion.npz")

    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.registry_name:
            import wandb
            api = wandb.Api()
            reg_name = args_cli.registry_name
            if ":" not in reg_name:
                reg_name += ":latest"
            artifact = api.artifact(reg_name)
            env_cfg.commands.motion.motion_file = str(Path(artifact.download()) / "motion.npz")

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)
    log_dir = os.path.dirname(resume_path)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        env.unwrapped, ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir, filename="policy.onnx",
    )
    attach_onnx_metadata(
        env.unwrapped,
        args_cli.wandb_path if args_cli.wandb_path else "none",
        export_model_dir,
    )

    # ── Success tracker ────────────────────────────────────────────────────
    num_envs = env.unwrapped.num_envs
    tracker  = BackflipSuccessTracker(num_envs)
    target_episodes = args_cli.eval_episodes
    print(f"\n[EVAL] Evaluating success rate over {target_episodes} episodes "
          f"across {num_envs} parallel envs...\n")

    obs, _   = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions      = policy(obs)
            obs, _, dones, extras = env.step(actions)

        # Extract base quaternion and projected gravity from env
        robot             = env.unwrapped.scene["robot"]
        base_quat_w       = robot.data.root_quat_w          # (N,4) [w,x,y,z]
        projected_gravity = robot.data.projected_gravity_b  # (N,3)
        timed_out         = extras.get("time_outs", None)

        tracker.update(base_quat_w, projected_gravity)
        tracker.register_episode_end(dones, timed_out)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # Stop once we have enough episodes
        if tracker.completed >= target_episodes:
            break

    tracker.summarize()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()