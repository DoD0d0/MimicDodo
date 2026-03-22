"""Convert dodo_daimao backflip CSV to NPZ format using Pinocchio FK.
No Isaac Sim needed — runs in trajopt conda env in <5 seconds.

Usage:
    conda activate trajopt
    python scripts/csv_to_npz_dodo_daimao.py \
        --input_file scripts/dodo_daimao_backflip.csv \
        --input_fps 100 \
        --output_name dodo_daimao_backflip \
        --output_fps 100
"""
import argparse
import sys
import os
import shutil
from pathlib import Path
import numpy as np

# Allow running from whole_body_tracking/ or MimicDodo/
script_dir   = Path(__file__).resolve().parent
mimic_root   = script_dir.parent
trajopt_src  = mimic_root / "trajectory_creation" / "se3_trajopt" / "src"
sys.path.insert(0, str(trajopt_src))

import pinocchio as pin
from robots.dodo_daimao.DodoDaimaoWrapper import DodoDaimao

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_file",  type=str, required=True)
parser.add_argument("--input_fps",   type=int, default=100)
parser.add_argument("--output_name", type=str, required=True)
parser.add_argument("--output_fps",  type=int, default=100)
parser.add_argument("--frame_range", nargs=2, type=int, metavar=("START","END"), default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load robot
# ---------------------------------------------------------------------------
robot = DodoDaimao()
model = robot.model
data  = robot.data

# Body names expected by the training task (must match flat_env_cfg body_names)
BODY_NAMES = [
    "body",
    "hip_left",       "upper_leg_left",  "lower_leg_left",  "foot_left",
    "hip_right",      "upper_leg_right", "lower_leg_right", "foot_right",
]

body_ids = [model.getFrameId(n, pin.FrameType.BODY) for n in BODY_NAMES]
print(f"Tracking {len(body_ids)} bodies: {BODY_NAMES}")

# Joint save order — must match actuator order in dodo_daimao.py robot config
JOINT_SAVE_ORDER = [
    "hip_left",       "upper_leg_left",  "lower_leg_left",  "foot_left",
    "hip_right",      "upper_leg_right", "lower_leg_right", "foot_right",
]
joint_q_ids = []
joint_v_ids = []
for name in JOINT_SAVE_ORDER:
    jid = model.getJointId(name)
    joint_q_ids.append(model.joints[jid].idx_q)
    joint_v_ids.append(model.joints[jid].idx_v)

# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------
print(f"Loading {args.input_file} ...")
if args.frame_range:
    raw = np.loadtxt(args.input_file, delimiter=",",
                     skiprows=args.frame_range[0]-1,
                     max_rows=args.frame_range[1]-args.frame_range[0]+1)
else:
    raw = np.loadtxt(args.input_file, delimiter=",")

raw = raw.astype(np.float32)
print(f"  CSV shape: {raw.shape}  (frames x cols)")

# CSV column layout written by backflip script:
#   0:3   base position (x,y,z)
#   3:7   base quaternion (qx,qy,qz,qw)  <- pinocchio convention
#   7:15  joint positions (JOINT_SAVE_ORDER)
assert raw.shape[1] >= 15, f"Expected >=15 cols, got {raw.shape[1]}"

base_pos_in  = raw[:, 0:3]   # (F,3)
base_quat_in = raw[:, 3:7]   # (F,4) pinocchio [qx,qy,qz,qw]
dof_pos_in   = raw[:, 7:15]  # (F,8)
F_in = raw.shape[0]

# ---------------------------------------------------------------------------
# Interpolate to output fps
# ---------------------------------------------------------------------------
input_dt  = 1.0 / args.input_fps
output_dt = 1.0 / args.output_fps
duration  = (F_in - 1) * input_dt
times     = np.arange(0, duration, output_dt, dtype=np.float32)
F_out     = len(times)

def lerp(a, b, t):
    return a * (1 - t) + b * t

def slerp_quat(q0, q1, t):
    """Slerp between two quaternions [qx,qy,qz,qw]."""
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return lerp(q0, q1, t) / np.linalg.norm(lerp(q0, q1, t))
    theta0    = np.arccos(dot)
    theta     = theta0 * t
    sin_theta = np.sin(theta)
    sin_theta0= np.sin(theta0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return s0 * q0 + s1 * q1

phase   = times / duration
idx0    = np.floor(phase * (F_in - 1)).astype(int)
idx1    = np.minimum(idx0 + 1, F_in - 1)
blend   = phase * (F_in - 1) - idx0

base_pos  = lerp(base_pos_in[idx0],  base_pos_in[idx1],  blend[:,None])
dof_pos   = lerp(dof_pos_in[idx0],   dof_pos_in[idx1],   blend[:,None])
base_quat = np.stack([slerp_quat(base_quat_in[i0], base_quat_in[i1], b)
                      for i0, i1, b in zip(idx0, idx1, blend)])

print(f"  Interpolated: {F_in} frames @ {args.input_fps}Hz -> {F_out} frames @ {args.output_fps}Hz")

# ---------------------------------------------------------------------------
# Compute velocities (finite differences)
# ---------------------------------------------------------------------------
def finite_diff(x, dt):
    """Central differences with forward/backward at edges."""
    vel = np.zeros_like(x)
    vel[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    vel[0]    = (x[1]  - x[0])   / dt
    vel[-1]   = (x[-1] - x[-2])  / dt
    return vel

base_lin_vel = finite_diff(base_pos,  output_dt)   # (F,3)
dof_vel      = finite_diff(dof_pos,   output_dt)   # (F,8)

# Angular velocity from quaternion finite differences -> axis-angle / dt
def quat_ang_vel(quats, dt):
    """Compute body-frame angular velocity from quaternion sequence."""
    ang_vels = np.zeros((len(quats), 3), dtype=np.float32)
    for i in range(1, len(quats) - 1):
        q_prev = quats[i-1]  # [qx,qy,qz,qw]
        q_next = quats[i+1]
        # q_rel = q_next * conj(q_prev)
        qp = np.array([-q_prev[0], -q_prev[1], -q_prev[2], q_prev[3]])  # conjugate
        # Hamilton product q_next * qp
        x0,y0,z0,w0 = q_next
        x1,y1,z1,w1 = qp
        qr = np.array([
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1,
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
        ])
        # axis-angle from quaternion
        vec  = qr[:3]
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            angle = 2.0 * np.arctan2(norm, qr[3])
            axis  = vec / norm
            ang_vels[i] = axis * angle / (2.0 * dt)
    ang_vels[0]  = ang_vels[1]
    ang_vels[-1] = ang_vels[-2]
    return ang_vels

base_ang_vel = quat_ang_vel(base_quat, output_dt)  # (F,3)

# ---------------------------------------------------------------------------
# Convert quaternion: pinocchio [qx,qy,qz,qw] -> Isaac [qw,qx,qy,qz]
# ---------------------------------------------------------------------------
isaac_quat = base_quat[:, [3, 0, 1, 2]]   # (F,4) wxyz

# ---------------------------------------------------------------------------
# Run FK for each frame to get body world positions & orientations
# ---------------------------------------------------------------------------
print("Running Pinocchio FK for all frames ...")
n_bodies  = len(body_ids)
body_pos  = np.zeros((F_out, n_bodies, 3),  dtype=np.float32)
body_quat = np.zeros((F_out, n_bodies, 4),  dtype=np.float32)  # wxyz for Isaac

# Build full pinocchio q vector for each frame
q_full = np.zeros((F_out, model.nq), dtype=np.float64)
q_full[:, 0:3] = base_pos
# pinocchio stores quat as [qx,qy,qz,qw] at indices 3:7
q_full[:, 3:7] = base_quat
for i, q_id in enumerate(joint_q_ids):
    q_full[:, q_id] = dof_pos[:, i]

for f in range(F_out):
    pin.forwardKinematics(model, data, q_full[f])
    pin.updateFramePlacements(model, data)
    for b, fid in enumerate(body_ids):
        T = data.oMf[fid]
        body_pos[f, b]  = T.translation.astype(np.float32)
        # pinocchio rotation -> quaternion [qw,qx,qy,qz] for Isaac
        r  = pin.Quaternion(T.rotation)
        body_quat[f, b] = np.array([r.w, r.x, r.y, r.z], dtype=np.float32)

print(f"  FK done. body_pos shape: {body_pos.shape}")

# Body velocities
body_lin_vel = finite_diff(body_pos.reshape(F_out, -1), output_dt).reshape(F_out, n_bodies, 3)
body_ang_vel = np.zeros((F_out, n_bodies, 3), dtype=np.float32)  # approx zero for ref motion

# ---------------------------------------------------------------------------
# Package log dict matching original csv_to_npz format
# ---------------------------------------------------------------------------
log = {
    "fps":            np.array([args.output_fps]),
    "joint_pos":      dof_pos.astype(np.float32),            # (F,8)
    "joint_vel":      dof_vel.astype(np.float32),            # (F,8)
    "body_pos_w":     body_pos,                              # (F,B,3)
    "body_quat_w":    body_quat,                             # (F,B,4) wxyz
    "body_lin_vel_w": body_lin_vel.astype(np.float32),       # (F,B,3)
    "body_ang_vel_w": body_ang_vel,                          # (F,B,3)
}

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_name = args.output_name
if not output_name.endswith(".npz"):
    output_name += ".npz"

# Save next to the script (whole_body_tracking/) so train.py finds it
out_path = os.path.join(str(script_dir.parent), output_name)
np.savez(out_path, **log)
print(f"\n[INFO] Saved: {out_path}")
print(f"  joint_pos  : {log['joint_pos'].shape}")
print(f"  body_pos_w : {log['body_pos_w'].shape}  bodies={BODY_NAMES}")
print(f"  fps        : {args.output_fps}")

# Also try to upload to wandb if available
try:
    import wandb
    run = wandb.init(project="csv_to_npz", name=args.output_name)
    artifact = run.log_artifact(out_path, name=args.output_name, type="motions")
    run.link_artifact(artifact, target_path=f"wandb-registry-motions/{args.output_name}")
    print(f"[INFO] Uploaded to wandb registry: motions/{args.output_name}")
    run.finish()
except ImportError:
    print("[INFO] wandb not available in this env — skipping upload")
except Exception as e:
    print(f"[INFO] wandb upload skipped: {e}")