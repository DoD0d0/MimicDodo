import pinocchio as pin
import numpy as np
import sys
import os
import time
from pinocchio.visualize import MeshcatVisualizer

# --- 1. CONFIGURATION ---
# Calculate paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "../..")
urdf_filename = os.path.join(project_root, "assets/dodo/urdf/dodo.urdf")
# package_dirs should point to parent of 'dodo' package since URDF uses package://dodo/meshes/...
mesh_dir = os.path.join(project_root, "assets/")
OUTPUT_CSV = os.path.join("whole_body_tracking", "scripts", "dodo_walk.csv")

INITIAL_BASE_HEIGHT = 0.55                  

# Walk Parameters
STEP_LENGTH = 0.15
STEP_HEIGHT = 0.05
STEP_FREQ   = 1.5
DT          = 0.01 
FORWARD_SPEED = STEP_FREQ * STEP_LENGTH 
RECORD_DURATION = 5.0 

# Default Crouch (for IK seeding)
initial_pose_rad = [0.0, 0.4, -0.7, 0.3]

# 🔴 CSV EXPORT ORDER (Block: Left then Right)
# This MUST match the 'joint_names' list in your csv_to_npz_dodo.py
CSV_EXPORT_ORDER = [
    'left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4',
    'right_joint_1', 'right_joint_2', 'right_joint_3', 'right_joint_4'
]

# --- 2. SETUP ---
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_filename, package_dirs=mesh_dir, root_joint=pin.JointModelFreeFlyer()
)
data = model.createData()

viz = MeshcatVisualizer(model, collision_model, visual_model)
try:
    viz.initViewer(open=True)
except ImportError:
    pass 
viz.loadViewerModel()

left_foot_id = model.getFrameId("left_link_4")
right_foot_id = model.getFrameId("right_link_4")

# --- 3. INITIALIZATION ---
q = pin.neutral(model) 

# Map the initial pose to the correct joints in Pinocchio's q
# (We don't know if q is interleaved or block, so we map by name)
joint_map_init = {
    'left_joint_1': initial_pose_rad[0], 'right_joint_1': initial_pose_rad[0],
    'left_joint_2': initial_pose_rad[1], 'right_joint_2': initial_pose_rad[1],
    'left_joint_3': initial_pose_rad[2], 'right_joint_3': initial_pose_rad[2],
    'left_joint_4': initial_pose_rad[3], 'right_joint_4': initial_pose_rad[3]
}

for name, val in joint_map_init.items():
    if model.existJointName(name):
        idx = model.getJointId(name)
        q_idx = model.joints[idx].idx_q
        q[q_idx] = val

# Height Calculation
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

p_right_temp = data.oMf[right_foot_id].translation.copy()
p_left_temp  = data.oMf[left_foot_id].translation.copy()
min_foot_z = min(p_right_temp[2], p_left_temp[2])
FOOT_RADIUS = 0.035 
SAFETY_OFFSET = 0.00 
auto_height = -min_foot_z + FOOT_RADIUS + SAFETY_OFFSET
print(f"[INFO] Auto Height: {auto_height:.4f}m")

base_h = INITIAL_BASE_HEIGHT if INITIAL_BASE_HEIGHT is not None else auto_height
q[2] = base_h

# Update Start Positions
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
p_right_start_global = data.oMf[right_foot_id].translation.copy()
p_left_start_global  = data.oMf[left_foot_id].translation.copy()
ROT_FLAT = np.eye(3)

# --- 4. IK FUNCTION (6D) ---
def solve_ik_6d(model, data, frame_id, target_pos, target_rot, q_curr):
    dt_ik = 0.1
    damp = 1e-6 
    
    for i in range(20): 
        pin.forwardKinematics(model, data, q_curr)
        pin.updateFramePlacements(model, data)
        M_curr = data.oMf[frame_id]
        
        err_pos = target_pos - M_curr.translation
        err_rot_global = pin.log3(target_rot @ M_curr.rotation.T)
        err_full = np.concatenate([err_pos, err_rot_global])
        
        if np.linalg.norm(err_full) < 1e-4: 
            return q_curr
            
        J = pin.computeFrameJacobian(model, data, q_curr, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J[:, 0:6] = 0.0 # Don't move base
        
        v = J.T @ np.linalg.inv(J @ J.T + damp * np.eye(6)) @ err_full
        q_curr = pin.integrate(model, q_curr, v * dt_ik)
        
    return q_curr

# --- 5. RECORDING LOOP ---
print(f"Recording walk for {RECORD_DURATION}s...")
recorded_rows = [] 

t = 0.0
while t < RECORD_DURATION:
    t += DT
    
    # 1. BODY
    body_x_pos = t * FORWARD_SPEED
    q[0] = body_x_pos
    q[2] = base_h 
    
    # 2. FEET TRAJECTORY
    phase = (t * STEP_FREQ) * 2 * np.pi
    
    # Right Leg
    x_cyc = -np.cos(phase) * (STEP_LENGTH / 2)
    z_cyc = np.sin(phase) * STEP_HEIGHT if np.sin(phase) > 0 else 0.0
    target_right_pos = p_right_start_global.copy()
    target_right_pos[0] += body_x_pos + x_cyc
    target_right_pos[2] = p_right_start_global[2] + z_cyc 
    
    # Left Leg
    phase_l = (phase + np.pi) % (2 * np.pi)
    x_cyc_l = -np.cos(phase_l) * (STEP_LENGTH / 2)
    z_cyc_l = np.sin(phase_l) * STEP_HEIGHT if np.sin(phase_l) > 0 else 0.0
    target_left_pos = p_left_start_global.copy()
    target_left_pos[0] += body_x_pos + x_cyc_l
    target_left_pos[2] = p_left_start_global[2] + z_cyc_l
    
    # 3. SOLVE IK
    q = solve_ik_6d(model, data, right_foot_id, target_right_pos, ROT_FLAT, q)
    q = solve_ik_6d(model, data, left_foot_id, target_left_pos, ROT_FLAT, q)
    
    # 4. EXPORT PREPARATION (RE-ORDERING)
    # Extract joints from Pinocchio's q and place them in CSV Block Order
    joint_values_block = []
    
    for name in CSV_EXPORT_ORDER:
        idx = model.getJointId(name)
        q_idx = model.joints[idx].idx_q
        joint_values_block.append(q[q_idx])
    
    # Row: [Pos(3), Quat(4), Joints(8)]
    # Note: We use 'joint_values_block', NOT q[7:15] directly!
    row = np.concatenate([
        q[0:3], 
        q[3:7], 
        np.array(joint_values_block)
    ])
    
    viz.display(q)
    recorded_rows.append(row)
    
    time.sleep(DT)

# --- 6. SAVE ---
output_dir = os.path.dirname(OUTPUT_CSV)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Saving {len(recorded_rows)} frames to {OUTPUT_CSV}...")
print("Format: Left Legs (Cols 7-10) then Right Legs (Cols 11-14)")
np.savetxt(OUTPUT_CSV, np.array(recorded_rows), delimiter=",")
print("Done.")