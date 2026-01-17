import pinocchio as pin
import numpy as np
import os
import time
from pinocchio.visualize import MeshcatVisualizer

# =============================================================================
# CONFIG
# =============================================================================

# Calculate paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "../..")
URDF = os.path.join(project_root, "assets/dodo/urdf/dodo.urdf")
# package_dirs should point to parent of 'dodo' package since URDF uses package://dodo/meshes/...
MESH_DIR = os.path.join(project_root, "assets/")
OUTPUT_CSV = os.path.join("whole_body_tracking", "scripts", "dodo_kangoroo.csv")

DT = 0.02

NUM_JUMPS     = 10
JUMP_DISTANCE = 0.25
JUMP_HEIGHT   = 0.15
JUMP_TIME     = 1.0

# LEG MOTION AMPLITUDES (RAD) — KEEP SMALL
HIP_PITCH_AMP = np.deg2rad(35)   # link_2
KNEE_AMP      = np.deg2rad(45)   # link_3
ANKLE_AMP     = np.deg2rad(1)   # link_4

INITIAL_POSE_DEG = [0.0, 25.0, -35.0, 10.0]

# =============================================================================
# SETUP
# =============================================================================

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    URDF, package_dirs=MESH_DIR, root_joint=pin.JointModelFreeFlyer()
)
data = model.createData()

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# =============================================================================
# INITIAL STATE
# =============================================================================

q = pin.neutral(model)

pose_rad = np.deg2rad(INITIAL_POSE_DEG)
q[7:11]  = pose_rad
q[11:15] = pose_rad

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# Compute standing height
left_foot_id  = model.getFrameId("left_link_4")
right_foot_id = model.getFrameId("right_link_4")

p_l = data.oMf[left_foot_id].translation
p_r = data.oMf[right_foot_id].translation

FOOT_RADIUS = 0.035
base_height = -min(p_l[2], p_r[2]) + FOOT_RADIUS
q[2] = base_height

# Store neutral joint angles
q0 = q.copy()

# =============================================================================
# HELPERS
# =============================================================================

def smoothstep(s):
    return 0.5 * (1 - np.cos(np.pi * s))

# =============================================================================
# MAIN LOOP
# =============================================================================

recorded = []
t = 0.0
total_time = NUM_JUMPS * JUMP_TIME

print("[INFO] Recording CLEAN kangaroo jumps (correct joints)...")

while t <= total_time:

    jump_idx = int(t // JUMP_TIME)
    local_t  = t % JUMP_TIME
    s = local_t / JUMP_TIME  # 0 → 1

    # ---------------- BASE ----------------
    q[0] = jump_idx * JUMP_DISTANCE + JUMP_DISTANCE * smoothstep(s)
    q[2] = base_height + JUMP_HEIGHT * np.sin(np.pi * s)

    # ---------------- LEGS ----------------
    hip_pitch =  HIP_PITCH_AMP * np.sin(np.pi * s)
    knee      = -KNEE_AMP      * np.sin(np.pi * s)   # ← IMPORTANT
    ankle     =  ANKLE_AMP     * np.sin(np.pi * s)

    # LEFT LEG
    q[7]  = q0[7]             # link_1 (hip yaw) — FIXED
    q[8]  = q0[8] + hip_pitch # link_2
    q[9]  = q0[9] + knee      # link_3
    q[10] = q0[10] + ankle    # link_4

    # RIGHT LEG
    q[11] = q0[11]
    q[12] = q0[12] + hip_pitch
    q[13] = q0[13] + knee
    q[14] = q0[14] + ankle

    viz.display(q)
    recorded.append(q.copy())

    t += DT
    time.sleep(DT)

# =============================================================================
# EXPORT
# =============================================================================

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
np.savetxt(OUTPUT_CSV, np.array(recorded), delimiter=",")

print(f"[DONE] Saved {len(recorded)} frames → {OUTPUT_CSV}")
