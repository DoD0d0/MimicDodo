import pinocchio as pin
import numpy as np
import sys
import os
import time
from pinocchio.visualize import MeshcatVisualizer

# --- 1. KONFIGURATION ---
# Calculate paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "../..")
urdf_filename = os.path.join(project_root, "assets/dodo/urdf/dodo.urdf")
# package_dirs should point to parent of 'dodo' package since URDF uses package://dodo/meshes/...
mesh_dir = os.path.join(project_root, "assets/")
OUTPUT_CSV = os.path.join("whole_body_tracking", "scripts", "dodo_jump.csv")

# --- JUMP PARAMETERS (VARIABLE SCOPE) ---
JUMP_DISTANCE   = 0.5   # How far forward to jump (meters)
JUMP_HEIGHT_ADD = 0.20  # How high to jump above standing height (meters)
TUCK_HEIGHT     = 0.05  # How much to lift feet in the air (meters)

# --- TIMING ---
DT              = 0.02  # 50 Hz
T_PREPARE       = 0.5   # Stand still
T_CROUCH        = 0.5   # Dip down
T_AIR           = 0.6   # Time in the air
T_RECOVER       = 0.8   # Land and stand up
T_HOLD          = 0.5   # Hold finish pose

TOTAL_TIME = T_PREPARE + T_CROUCH + T_AIR + T_RECOVER + T_HOLD

# Start Angles
initial_pose_deg = [0.0, 25.0, -35.0, 10.0] 

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
pose_rad = [np.deg2rad(x) for x in initial_pose_deg]
q[7:11]  = pose_rad
q[11:15] = pose_rad

# Calculate Standing Height (0.0 offset)
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

p_right_temp = data.oMf[right_foot_id].translation.copy()
p_left_temp  = data.oMf[left_foot_id].translation.copy()
min_foot_z = min(p_right_temp[2], p_left_temp[2])
FOOT_RADIUS = 0.035 
base_standing_h = -min_foot_z + FOOT_RADIUS + 0.00 # Exact ground contact

q[2] = base_standing_h
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# Initial Foot Positions
p_right_start = data.oMf[right_foot_id].translation.copy()
p_left_start  = data.oMf[left_foot_id].translation.copy()
ROT_FLAT = np.eye(3)

# --- 4. HELPER: 6D IK SOLVER ---
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
        
        if np.linalg.norm(err_full) < 1e-4: return q_curr
            
        J = pin.computeFrameJacobian(model, data, q_curr, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J[:, 0:6] = 0.0 
        v = J.T @ np.linalg.inv(J @ J.T + damp * np.eye(6)) @ err_full
        q_curr = pin.integrate(model, q_curr, v * dt_ik)
    return q_curr

# --- 5. HELPER: TRAJECTORY GENERATORS ---
def get_parabola(t, t_start, t_end, start_val, end_val, peak_add):
    """ Returns value for a parabolic arc (Start -> Peak -> End) """
    duration = t_end - t_start
    if duration <= 0: return end_val
    norm_t = (t - t_start) / duration # 0..1
    
    # Linear part
    linear = start_val + (end_val - start_val) * norm_t
    # Parabolic hump (4 * x * (1-x) creates a hump 0->1->0)
    hump = peak_add * 4 * norm_t * (1 - norm_t)
    return linear + hump

def get_linear(t, t_start, t_end, start_val, end_val):
    """ Simple linear interpolation """
    if t < t_start: return start_val
    if t > t_end: return end_val
    norm_t = (t - t_start) / (t_end - t_start)
    return start_val + (end_val - start_val) * norm_t

def get_cosine_interp(t, t_start, t_end, start_val, end_val):
    """ Smoother S-Curve interpolation """
    if t < t_start: return start_val
    if t > t_end: return end_val
    norm_t = (t - t_start) / (t_end - t_start)
    # Cosine map 0..1 -> 0..1
    smooth_t = (1 - np.cos(norm_t * np.pi)) / 2
    return start_val + (end_val - start_val) * smooth_t

# --- 6. RECORDING LOOP ---
print(f"[INFO] Starting Jump Recording...")
print(f"       Distance: {JUMP_DISTANCE}m, Height: {JUMP_HEIGHT_ADD}m")
recorded_data = [] 

t = 0.0
# Define timing milestones
t1 = T_PREPARE
t2 = t1 + T_CROUCH
t3 = t2 + T_AIR
t4 = t3 + T_RECOVER

while t <= TOTAL_TIME:
    
    # --- A. BASE TRAJECTORY ---
    curr_base_x = 0.0
    curr_base_z = base_standing_h
    
    if t < t1:
        # 1. PREPARE: Stand still
        pass
        
    elif t < t2:
        # 2. CROUCH: Dip down slightly
        # Dip to 80% of height
        curr_base_z = get_cosine_interp(t, t1, t2, base_standing_h, base_standing_h * 0.7)
        curr_base_x = 0.0
        
    elif t < t3:
        # 3. AIR: Jump Parabola
        # X: Move from 0 to Distance
        curr_base_x = get_linear(t, t2, t3, 0.0, JUMP_DISTANCE)
        # Z: Start(Crouch) -> Peak(Stand+Add) -> End(Crouch)
        crouch_h = base_standing_h * 0.7
        # We want the peak to be (Standing + Add) relative to Crouch
        total_peak = (base_standing_h + JUMP_HEIGHT_ADD) - crouch_h
        curr_base_z = get_parabola(t, t2, t3, crouch_h, crouch_h, total_peak)
        
    elif t < t4:
        # 4. RECOVER: Stand up from crouch
        curr_base_x = JUMP_DISTANCE
        curr_base_z = get_cosine_interp(t, t3, t4, base_standing_h * 0.7, base_standing_h)
        
    else:
        # 5. HOLD
        curr_base_x = JUMP_DISTANCE
        curr_base_z = base_standing_h

    # Apply Base
    q[0] = curr_base_x
    q[2] = curr_base_z
    
    # --- B. FOOT TRAJECTORY ---
    # Feet stay at 0 until Jump, then move to Distance, then stay.
    
    curr_foot_x_r = p_right_start[0]
    curr_foot_x_l = p_left_start[0]
    
    curr_foot_z = p_right_start[2] # Ground level
    
    if t > t2 and t < t3:
        # In Air: Move feet forward + Tuck
        # X Motion
        progress = (t - t2) / T_AIR
        curr_foot_x_r = p_right_start[0] + (progress * JUMP_DISTANCE)
        curr_foot_x_l = p_left_start[0]  + (progress * JUMP_DISTANCE)
        
        # Z Motion (Tuck)
        # Parabola 0 -> Tuck -> 0
        curr_foot_z = p_right_start[2] + (TUCK_HEIGHT * 4 * progress * (1 - progress))
        
    elif t >= t3:
        # Landed
        curr_foot_x_r = p_right_start[0] + JUMP_DISTANCE
        curr_foot_x_l = p_left_start[0]  + JUMP_DISTANCE
        curr_foot_z = p_right_start[2]

    # Target Vectors
    target_r = np.array([curr_foot_x_r, p_right_start[1], curr_foot_z])
    target_l = np.array([curr_foot_x_l, p_left_start[1], curr_foot_z])
    
    # --- C. SOLVE IK ---
    # Use Rotation Lock to keep feet flat
    q = solve_ik_6d(model, data, right_foot_id, target_r, ROT_FLAT, q)
    q = solve_ik_6d(model, data, left_foot_id, target_l, ROT_FLAT, q)
    
    # --- D. SAVE ---
    viz.display(q)
    recorded_data.append(q.copy())
    
    t += DT
    time.sleep(DT)

# --- 7. EXPORT ---
output_dir = os.path.dirname(OUTPUT_CSV)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Saving {len(recorded_data)} frames to {OUTPUT_CSV}...")
np.savetxt(OUTPUT_CSV, np.array(recorded_data), delimiter=",")
print("Done!")