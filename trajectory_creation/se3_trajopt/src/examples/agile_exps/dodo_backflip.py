import numpy as np
import time
import pinocchio as pin

from nltrajopt.trajectory_optimization import NLTrajOpt
from nltrajopt.contact_scheduler import ContactScheduler
from nltrajopt.node import Node
from nltrajopt.constraint_models import *
from nltrajopt.cost_models import *
import nltrajopt.utils as reprutils
from visualiser.visualiser import TrajoptVisualiser

# --- IMPORT YOUR NEW WRAPPER ---
from robots.dodobot_v3.DodoWrapper import Dodo

np.set_printoptions(precision=2, suppress=True)

import nltrajopt.params as pars

VIS = pars.VIS
DT = 0.05

# 1. LOAD DODO
robot = Dodo()
q = robot.go_neutral()

# 2. SETUP TERRAIN (Standard)
from terrain.terrain_grid import TerrainGrid
terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
terrain.set_zero()

contacts_dict = {
    "l_foot": robot.left_foot_frames,
    "r_foot": robot.right_foot_frames,
}

# 3. SCHEDULE (Standard Jump Sequence)
contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)
contact_scheduler.add_phase(["l_foot", "r_foot"], 1.0) # Stand
k1 = len(contact_scheduler.contact_sequence_fnames)
contact_scheduler.add_phase([], 0.45)                  # Flight (increased slightly for safety)
k2 = len(contact_scheduler.contact_sequence_fnames)
contact_scheduler.add_phase(["l_foot", "r_foot"], 1.0) # Land

frame_contact_seq = contact_scheduler.contact_sequence_fnames
contact_frame_names = robot.left_foot_frames + robot.right_foot_frames

# 4. DEFINE COST WEIGHTS FOR JOINTS
# Dodo has 8 joints (excluding base).
# Order in 'v' vector (velocity): [Right_J1, Right_J2, Right_J3, Right_J4, Left_J1, Left_J2, Left_J3, Left_J4]
# J1 is the sideways hip. We want to penalized this heavily.
# J2, J3, J4 are the flipping joints. We want low cost.

# Create a diagonal weight matrix
w_diag = np.array([
    10.0,  0.1, 0.1, 0.1,  # Right Leg: High cost on J1, Low on J2-J4
    10.0,  0.1, 0.1, 0.1   # Left Leg:  High cost on J1, Low on J2-J4
])
weights_mat = np.diag(w_diag)


stages = []
K = len(frame_contact_seq)
print("K = ", K)

for k, contact_phase_fnames in enumerate(frame_contact_seq):
    stage_node = Node(
        nv=robot.model.nv,
        contact_phase_fnames=contact_phase_fnames,
        contact_fnames=contact_frame_names,
    )

    dyn_const = WholeBodyDynamics()
    stage_node.dynamics_type = dyn_const.name

    stage_node.constraints_list.extend(
        [
            dyn_const,
            TimeConstraint(min_dt=DT, max_dt=DT, total_time=None),
            SemiEulerIntegration(),
            TerrainGridContactConstraints(terrain),
            # Reduced friction constraint slightly as point feet slip easier
            TerrainGridFrictionConstraints(terrain, max_delta_force=20), 
        ]
    )

    # COST 1: Stay close to neutral pose (keeps hips straight)
    stage_node.costs_list.extend([ConfigurationCost(q.copy()[7:], np.eye(robot.model.nq - 7) * 1e-3)])
    
    # COST 2: Joint Acceleration - USE OUR CUSTOM WEIGHTS
    # We multiply our custom matrix by a scaling factor (e.g., 1e-4)
    stage_node.costs_list.extend([JointAccelerationCost(
        np.zeros((robot.model.nv - 6,)), 
        weights_mat * 1e-4 
    )])

    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

# 5. INITIAL & TARGET POSE
opti.set_initial_pose(q)

qf = np.copy(q)
qf[0] = -0.5 # Move back 0.5m
# For Dodo, we might need a higher Z target if the legs are short?
# qf[2] = q[2] # Keep height same as start
opti.set_target_pose(qf)

# 6. WARM START (The Flip Rotation)
for k, node in enumerate(opti.nodes):
    if k1 <= k <= k2:
        # Interpolate rotation from 0 to -2pi (backflip)
        theta = -2 * np.pi * (k - k1) / (k2 - k1)
        # Apply pitch rotation (Y-axis) to the base
        opti.x0[node.q_id] = reprutils.rpy2rep(q, [0.0, theta, 0.0])

# 7. SOLVE
print("Solving...")
result = opti.solve(300, 1e-3, False, print_level=1) # Increased iterations slightly
opti.save_solution("dodo_flip")


# 8. VISUALIZE
K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

if VIS:
    tvis = TrajoptVisualiser(robot)
    tvis.display_robot_q(robot, qs[0])
    time.sleep(1)
    
    # ---------------------------------------------------------
    #                 EXPORT TO CSV (100Hz UPSAMPLING)
    # ---------------------------------------------------------
    import os

    # 1. Define Output Path
    output_path = os.path.join("whole_body_tracking", "scripts", "dodo_backflip.csv")

    TARGET_FREQ = 100.0  # We want 100Hz for training
    SOLVER_DT   = DT     # Currently 0.05s

    # 2. Define Joint Order (Matches training script)
    CSV_EXPORT_ORDER = [
        'left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4',
        'right_joint_1', 'right_joint_2', 'right_joint_3', 'right_joint_4'
    ]

    # 3. Pre-calculate indices
    joint_indices = []
    for name in CSV_EXPORT_ORDER:
        if robot.model.existJointName(name):
            j_id = robot.model.getJointId(name)
            q_idx = robot.model.joints[j_id].idx_q
            joint_indices.append(q_idx)

    # 4. UPSAMPLING LOOP
    # We use Pinocchio to interpolate between the solver's coarse frames
    print(f"\n[Export] Upsampling solution from {1/SOLVER_DT:.0f}Hz to {TARGET_FREQ:.0f}Hz...")

    export_rows = []
    steps_per_frame = int(SOLVER_DT * TARGET_FREQ) # e.g. 0.05 * 100 = 5 steps

    for k in range(len(qs) - 1):
        q_start = qs[k]
        q_end   = qs[k+1]
        
        # Generate intermediate frames
        for j in range(steps_per_frame):
            # alpha goes from 0.0 to 0.8 (in steps of 0.2)
            alpha = j / steps_per_frame
            
            # PINOCCHIO MAGIC: Handles Quaternion interpolation correctly!
            q_interp = pin.interpolate(robot.model, q_start, q_end, alpha)
            
            # --- Format Row ---
            base_pos  = q_interp[0:3]
            base_quat = q_interp[3:7]
            joint_vals = np.array([q_interp[idx] for idx in joint_indices])
            
            row = np.concatenate([base_pos, base_quat, joint_vals])
            export_rows.append(row)

    # Add the very last frame to finish the sequence
    q_final = qs[-1]
    row_final = np.concatenate([
        q_final[0:3], 
        q_final[3:7], 
        np.array([q_final[idx] for idx in joint_indices])
    ])
    export_rows.append(row_final)

    # 5. Save
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savetxt(output_path, np.array(export_rows), delimiter=",")
    print(f"[Export] Saved SMOOTH backflip to: {os.path.abspath(output_path)}")
    print(f"[Export] Original Frames: {len(qs)} -> Exported Frames: {len(export_rows)}")
    print("-" * 50)
    # ---------------------------------------------------------------------------------


    while True:
        for i in range(len(qs)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)
        tvis.update_forces(robot, {}, 0.01)