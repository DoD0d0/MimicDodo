"""Backflip trajectory optimization for DodoDaimao robot."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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

from robots.dodo_daimao.DodoDaimaoWrapper import DodoDaimao

np.set_printoptions(precision=2, suppress=True)

import nltrajopt.params as pars

VIS = pars.VIS
DT = 0.05

# 1. LOAD
robot = DodoDaimao()
q = robot.go_neutral()

# 2. TERRAIN
from terrain.terrain_grid import TerrainGrid
terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
terrain.set_zero()

contacts_dict = {
    "l_foot": robot.left_foot_frames,
    "r_foot": robot.right_foot_frames,
}

# 3. SCHEDULE
contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)
contact_scheduler.add_phase(["l_foot", "r_foot"], 1.0)
k1 = len(contact_scheduler.contact_sequence_fnames)
contact_scheduler.add_phase([], 0.45)
k2 = len(contact_scheduler.contact_sequence_fnames)
contact_scheduler.add_phase(["l_foot", "r_foot"], 1.0)

frame_contact_seq   = contact_scheduler.contact_sequence_fnames
contact_frame_names = robot.left_foot_frames + robot.right_foot_frames

print(f"K = {len(frame_contact_seq)}, flight steps = {k2-k1}")
assert k2 > k1

# 4. COST WEIGHTS
w_diag = np.array([
    1000.0, 0.1, 0.1, 0.1,
    1000.0, 0.1, 0.1, 0.1,
])
weights_mat = np.diag(w_diag)

w_hip_only = np.zeros((robot.model.nq - 7, robot.model.nq - 7))
w_hip_only[0, 0] = 1000.0   # hip_left
w_hip_only[4, 4] = 1000.0   # hip_right

# 5. BUILD STAGES
stages = []
for k, contact_phase_fnames in enumerate(frame_contact_seq):
    stage_node = Node(
        nv=robot.model.nv,
        contact_phase_fnames=contact_phase_fnames,
        contact_fnames=contact_frame_names,
    )

    dyn_const = WholeBodyDynamics()
    stage_node.dynamics_type = dyn_const.name

    stage_node.constraints_list.extend([
        dyn_const,
        TimeConstraint(min_dt=DT, max_dt=DT, total_time=None),
        SemiEulerIntegration(),
        TerrainGridContactConstraints(terrain),
        TerrainGridFrictionConstraints(terrain, max_delta_force=20),
    ])

    stage_node.costs_list.append(
        ConfigurationCost(q.copy()[7:], np.eye(robot.model.nq - 7) * 1e-3)
    )
    stage_node.costs_list.append(
        ConfigurationCost(q.copy()[7:], w_hip_only * 1e-2)
    )
    stage_node.costs_list.append(
        JointAccelerationCost(
            np.zeros((robot.model.nv - 6,)),
            weights_mat * 1e-4
        )
    )

    stages.append(stage_node)

# 6. OPTIMISER
opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)
opti.set_initial_pose(q)

qf = np.copy(q)
qf[0] = -0.5
opti.set_target_pose(qf)

# 7. WARM START
q_at_origin = np.copy(q)
q_at_origin[0] = 0.0
q_at_origin[1] = 0.0
q_at_origin[2] = 0.0

for k, node in enumerate(opti.nodes):
    if k1 <= k <= k2:
        theta = -2 * np.pi * (k - k1) / (k2 - k1)
        warm  = reprutils.rpy2rep(q_at_origin, [0.0, theta, 0.0])
        warm[0] = q[0]
        warm[1] = q[1]
        warm[2] = q[2]
        opti.x0[node.q_id] = warm

# 8. SOLVE
print("Solving...")
result = opti.solve(300, 1e-3, False, print_level=1)
opti.save_solution("dodo_daimao_flip")

# 9. UNPACK
K      = len(result["nodes"])
dts    = [result["nodes"][k]["dt"]     for k in range(K)]
qs     = [result["nodes"][k]["q"]      for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

if VIS:
    tvis = TrajoptVisualiser(robot)
    tvis.display_robot_q(robot, qs[0])
    time.sleep(1)

    output_path = os.path.join(
        "whole_body_tracking", "scripts", "dodo_daimao_backflip.csv"
    )
    TARGET_FREQ = 100.0
    SOLVER_DT   = DT

    CSV_EXPORT_ORDER = [
        "hip_left", "upper_leg_left", "lower_leg_left", "foot_left",
        "hip_right", "upper_leg_right", "lower_leg_right", "foot_right",
    ]

    joint_indices = []
    for name in CSV_EXPORT_ORDER:
        if robot.model.existJointName(name):
            j_id  = robot.model.getJointId(name)
            q_idx = robot.model.joints[j_id].idx_q
            joint_indices.append(q_idx)
        else:
            raise RuntimeError(f"Joint '{name}' not found.")

    steps_per_frame = int(SOLVER_DT * TARGET_FREQ)
    export_rows = []
    for k in range(len(qs) - 1):
        q_start, q_end = qs[k], qs[k + 1]
        for j in range(steps_per_frame):
            alpha    = j / steps_per_frame
            q_interp = pin.interpolate(robot.model, q_start, q_end, alpha)
            row = np.concatenate([
                q_interp[0:3], q_interp[3:7],
                np.array([q_interp[idx] for idx in joint_indices]),
            ])
            export_rows.append(row)

    q_final = qs[-1]
    export_rows.append(np.concatenate([
        q_final[0:3], q_final[3:7],
        np.array([q_final[idx] for idx in joint_indices]),
    ]))

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(output_path, np.array(export_rows), delimiter=",")
    print(f"[Export] Saved {len(export_rows)} rows → {os.path.abspath(output_path)}")

    while True:
        for i in range(len(qs)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)
        tvis.update_forces(robot, {}, 0.01)