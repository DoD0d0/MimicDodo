
import numpy as np
import os

# CONFIG
OUTPUT_FILE = os.path.join("whole_body_tracking", "scripts", "dodo_stand_simple.npz")
HEIGHT_STAND = 0.60

# 🔴 HARDCODED VALUES (No lists, no confusion)
VAL_HIP_ROLL  = 0.0
VAL_HIP_PITCH = 0.4
VAL_KNEE      = -0.7
VAL_ANKLE     = 0.3

# 🔴 ISAAC SIM INTERLEAVED ORDER
# Verify this list matches your "p self.robot.joint_names" exactly
ISAAC_ORDER = [
    'left_joint_1',  'right_joint_1', 
    'left_joint_2',  'right_joint_2', 
    'left_joint_3',  'right_joint_3', 
    'left_joint_4',  'right_joint_4'
]

# 🔴 EXPLICIT MAPPING
# Map Names to Variables directly.
joint_map = {
    'left_joint_1':  VAL_HIP_ROLL,
    'right_joint_1': VAL_HIP_ROLL,
    
    'left_joint_2':  VAL_HIP_PITCH,
    'right_joint_2': VAL_HIP_PITCH,
    
    'left_joint_3':  VAL_KNEE,
    'right_joint_3': VAL_KNEE,
    
    'left_joint_4':  VAL_ANKLE,
    'right_joint_4': VAL_ANKLE,
}

# GENERATE VECTOR
final_vector = np.zeros(8)
print("\n[INFO] Generated Vector Construction:")
for i, name in enumerate(ISAAC_ORDER):
    val = joint_map[name]
    final_vector[i] = val
    print(f"  Index {i} ({name}): {val}")

# SAVE
num_frames = 200
joint_pos_data = np.tile(final_vector, (num_frames, 1))
joint_vel_data = np.zeros((num_frames, 8))
body_pos_w     = np.zeros((num_frames, 1, 3)); body_pos_w[:, 0, :] = [0, 0, HEIGHT_STAND]
body_quat_w    = np.zeros((num_frames, 1, 4)); body_quat_w[:, 0, :] = [0, 0, 0, 1] # [x,y,z,w]
body_lin_vel_w = np.zeros((num_frames, 1, 3))
body_ang_vel_w = np.zeros((num_frames, 1, 3))

np.savez(OUTPUT_FILE, fps=50.0, joint_pos=joint_pos_data, joint_vel=joint_vel_data, 
         body_pos_w=body_pos_w, body_quat_w=body_quat_w, 
         body_lin_vel_w=body_lin_vel_w, body_ang_vel_w=body_ang_vel_w)
print(f"\nSaved to {OUTPUT_FILE}")
print(f"Expected Final Vector: [0.0, 0.0, 0.4, 0.4, -0.7, -0.7, 0.3, 0.3]")