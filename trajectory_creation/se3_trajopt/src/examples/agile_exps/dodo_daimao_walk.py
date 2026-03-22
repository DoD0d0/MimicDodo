"""Walking trajectory optimization for DodoDaimao robot."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import numpy as np
import pinocchio as pin

from nltrajopt.trajectory_optimization import NLTrajOpt
from nltrajopt.contact_scheduler import ContactScheduler
from nltrajopt.node import Node
from nltrajopt.constraint_models import *
from nltrajopt.cost_models import *
from terrain.terrain_grid import TerrainGrid
from visualiser.visualiser import TrajoptVisualiser

from robots.dodo_daimao.DodoDaimaoWrapper import DodoDaimao

import nltrajopt.params as pars

VIS = pars.VIS
DT = 0.05

robot = DodoDaimao()
q = robot.go_neutral()

terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
terrain.set_zero()

contacts_dict = {
    "l_foot": robot.left_foot_frames,
    "r_foot": robot.right_foot_frames,
}

contact_scheduler = ContactScheduler(robot.model, dt=DT, contact_frame_dict=contacts_dict)
contact_scheduler.add_phase(["l_foot", "r_foot"], 1.0)
for i in range(2):
    contact_scheduler.add_phase(["l_foot"],           0.4)
    contact_scheduler.add_phase(["l_foot", "r_foot"], 0.3)
    contact_scheduler.add_phase(["r_foot"],           0.4)
    contact_scheduler.add_phase(["l_foot", "r_foot"], 0.3)
contact_scheduler.add_phase(["l_foot", "r_foot"], 1.0)

frame_contact_seq   = contact_scheduler.contact_sequence_fnames
contact_frame_names = robot.left_foot_frames + robot.right_foot_frames

print(f"K = {len(frame_contact_seq)}")

stages = []
for contact_phase_fnames in frame_contact_seq:
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
        TerrainGridFrictionConstraints(terrain),
        TerrainGridContactConstraints(terrain),
    ])

    stage_node.costs_list.extend([
        ConfigurationCost(q.copy()[7:], np.eye(robot.model.nq - 7) * 1e-6)
    ])

    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)
opti.set_initial_pose(q)

qf = np.copy(q)
qf[0] = 0.5
opti.set_target_pose(qf)

print("Solving...")
result = opti.solve(300, 1e-3, parallel=False, print_level=1)
opti.save_solution("dodo_daimao_walk")

K      = len(result["nodes"])
dts    = [result["nodes"][k]["dt"]     for k in range(K)]
qs     = [result["nodes"][k]["q"]      for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

if VIS:
    tvis = TrajoptVisualiser(robot)
    tvis.display_robot_q(robot, qs[0])
    time.sleep(1)

    while True:
        for i in range(len(qs)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)
        tvis.update_forces(robot, {}, 0.01)