"""
Dodo Robot Configuration
Centralized configuration for the dodobot_v3 robot asset.
This config can be imported and used by all projects (trajectory_creation, rl_training, etc.)
"""
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Import asset directory
from . import ASSET_DIR

# =============================================================================
# SPAWN CONFIGURATION
# =============================================================================
DODO_SPAWN_CFG = sim_utils.UrdfFileCfg(
    asset_path=os.path.join(ASSET_DIR, "urdf/dodo.urdf"),
    activate_contact_sensors=True,
    fix_base=False,
    replace_cylinders_with_capsules=False,
    force_usd_conversion=True,
    joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,
            damping=0.0,
        ),
    ),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=None,
        max_depenetration_velocity=10.0,
        enable_gyroscopic_forces=True,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
        sleep_threshold=0.005,
        stabilization_threshold=0.001,
    ),
)

# =============================================================================
# ROBOT CONFIGURATION
# =============================================================================
DODO_ROBOT_CFG = ArticulationCfg(
    spawn=DODO_SPAWN_CFG,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),  # Perfect standing height
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_joint_1": 0.0,
            "left_joint_2": 0.4,
            "left_joint_3": -0.7,
            "left_joint_4": 0.3,
            "right_joint_1": 0.0,
            "right_joint_2": 0.4,
            "right_joint_3": -0.7,
            "right_joint_4": 0.3,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                'left_joint_1', 'right_joint_1',
                'left_joint_2', 'right_joint_2',
                'left_joint_3', 'right_joint_3',
                'left_joint_4', 'right_joint_4'
            ],
            effort_limit_sim=30.0,
            velocity_limit_sim=30.0,
            stiffness=40.0,
            damping=2.0
        ),
    },
)

