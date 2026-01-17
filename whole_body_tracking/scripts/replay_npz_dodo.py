"""Replay motion artifacts with the dodobot_v4 URDF.

This script has been simplified to ONLY load the 8-joint Dodo robot
and its corresponding 8-joint trajectory data.
"""

import argparse
import math
import numpy as np
import sys
from pathlib import Path

import torch

# Add the source directory to Python path so we can import whole_body_tracking
script_dir = Path(__file__).resolve().parent
source_dir = script_dir.parent / "source" / "whole_body_tracking"
sys.path.insert(0, str(source_dir))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions on the Dodo robot.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")
parser.add_argument(
    "--root_height_offset",
    type=float,
    default=0.0,
    help="Meters added to the root z position (useful if the URDF origin differs from the mocap data).",
)
parser.add_argument(
    "--root_roll_offset_deg",
    type=float,
    default=0.0,
    help="Roll offset (deg) applied to the root orientation before writing to sim.",
)
parser.add_argument(
    "--root_pitch_offset_deg",
    type=float,
    default=0.0,
    help="Pitch offset (deg) applied to the root orientation before writing to sim.",
)
parser.add_argument(
    "--root_yaw_offset_deg",
    type=float,
    default=0.0,
    help="Yaw offset (deg) applied to the root orientation before writing to sim.",
)
parser.add_argument(
    "--negate_joints",
    action="store_true",
    help="Negate all joint angles (useful if joint coordinate system is flipped).",
)
parser.add_argument(
    "--negate_left_joints",
    action="store_true",
    help="Negate left leg joint angles only.",
)
parser.add_argument(
    "--negate_right_joints",
    action="store_true",
    help="Negate right leg joint angles only.",
)
parser.add_argument(
    "--joint_offset_deg",
    nargs=8,
    type=float,
    default=None,
    metavar=("L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"),
    help="Add offset (degrees) to each joint: left_1, left_2, left_3, left_4, right_1, right_2, right_3, right_4",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_mul

##
# Pre-defined configs
##
from whole_body_tracking.robots.dodo import DODO_ROBOT_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


# --- ENTFERNT ---
# G1_JOINT_NAMES-Liste wurde entfernt.
# JOINT_NAME_MAP-Dictionary wurde entfernt.
# _build_source_index_tensor-Funktion wurde entfernt.
# Sie waren der Grund für den CUDA-Fehler.


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Returns a quaternion (w, x, y, z) for ZYX intrinsic rotations."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.tensor([w, x, y, z], dtype=torch.float32)


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = DODO_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    registry_name = args_cli.registry_name
    if ":" not in registry_name:
        registry_name += ":latest"
        
    import pathlib
    import os
    import glob
    import wandb

    api = wandb.Api()
    print(f"--- Downloading artifact: {registry_name} ---")
    artifact = api.artifact(registry_name)
    download_dir = artifact.download()
    
    # --- FIX START: Automatische Pfad-Korrektur (Colon vs Hyphen) ---
    if not os.path.exists(download_dir) and ":" in download_dir:
        # WandB meldet manchmal ":v1", erstellt aber "-v1" auf der Platte
        alt_dir = download_dir.replace(":", "-")
        if os.path.exists(alt_dir):
            print(f"[INFO] Pfad korrigiert: '{download_dir}' nicht gefunden, nutze '{alt_dir}'")
            download_dir = alt_dir
    # --- FIX ENDE ---

    print(f"Download directory: {download_dir}")
    
    # DEBUG: Zeige alle Dateien im Ordner an
    files_in_dir = glob.glob(f"{download_dir}/**/*", recursive=True)
    print(f"Files found: {files_in_dir}")

    # SUCHE nach der .npz Datei
    npz_files = [f for f in files_in_dir if f.endswith(".npz")]
    
    if not npz_files:
        # Fallback: Manchmal liegen die Dateien direkt im Root des Artifacts ohne Unterordner
        # Wir suchen nochmal rekursiv, falls der glob oben leer war
        files_in_dir = glob.glob(f"{download_dir}/*.npz")
        npz_files = [f for f in files_in_dir if f.endswith(".npz")]

    if not npz_files:
        raise FileNotFoundError(f"Keine .npz Datei im Ordner {download_dir} gefunden!")
    
    motion_file = npz_files[0]
    print(f"Using motion file: {motion_file}")


    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Orientation offset logic
    roll = math.radians(args_cli.root_roll_offset_deg)
    pitch = math.radians(args_cli.root_pitch_offset_deg)
    yaw = math.radians(args_cli.root_yaw_offset_deg)
    quat_offset = _quat_from_euler(roll, pitch, yaw).to(sim.device)
    apply_orientation_offset = not torch.allclose(
        quat_offset, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=sim.device), atol=1e-6
    )
    if apply_orientation_offset:
        quat_offset = quat_offset.unsqueeze(0)

    height_offset = args_cli.root_height_offset

    """
    # Joint Reihenfolge: Rechts, Links
    robot_joint_names = [
        "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
        "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
    ]
    """

    # replay_npz_dodo.py
    # WIR SAGEN DEM SKRIPT: "So ist das NPZ sortiert (Isaac Standard: Alphabetisch)!"
    robot_joint_names = [
        "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
        "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
    ]

    robot_joint_indices = robot.find_joints(robot_joint_names, preserve_order=True)[0]
    
    
    # Debug print once
    if time_steps[0] == 0:
        print(f"Motion joint shape: {motion.joint_pos.shape}")

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        if height_offset != 0.0:
            root_states[:, 2] += height_offset
        root_rot = motion.body_quat_w[time_steps][:, 0]
        if apply_orientation_offset:
            root_rot = quat_mul(quat_offset.repeat(root_rot.shape[0], 1), root_rot)
        root_states[:, 3:7] = root_rot
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)

        motion_joint_pos = motion.joint_pos[time_steps]
        motion_joint_vel = motion.joint_vel[time_steps]
        
        if args_cli.negate_joints:
            motion_joint_pos = -motion_joint_pos
            motion_joint_vel = -motion_joint_vel
        elif args_cli.negate_left_joints:
            motion_joint_pos[:, :4] = -motion_joint_pos[:, :4]
            motion_joint_vel[:, :4] = -motion_joint_vel[:, :4]
        elif args_cli.negate_right_joints:
            motion_joint_pos[:, 4:] = -motion_joint_pos[:, 4:]
            motion_joint_vel[:, 4:] = -motion_joint_vel[:, 4:]
        
        if args_cli.joint_offset_deg is not None:
            joint_offset_rad = torch.tensor(args_cli.joint_offset_deg, dtype=torch.float32, device=motion_joint_pos.device)
            joint_offset_rad = torch.deg2rad(joint_offset_rad)
            motion_joint_pos = motion_joint_pos + joint_offset_rad.unsqueeze(0)
        
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indices] = motion_joint_pos
        joint_vel[:, robot_joint_indices] = motion_joint_vel
        
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.01 # changed from 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()