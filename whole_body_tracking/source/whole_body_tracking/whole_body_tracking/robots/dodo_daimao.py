"""DodoDaimao robot configuration for Isaac Lab whole body tracking."""
import isaaclab.sim as sim_utils
import os
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "../../../../../"))
_ASSET_DIR = os.path.join(_project_root, "assets/dodo_daimao")

DODO_DAIMAO_SPAWN_CFG = sim_utils.UrdfFileCfg(
    asset_path=os.path.join(_ASSET_DIR, "urdf/dodo_daimao.urdf"),
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

DODO_DAIMAO_ROBOT_CFG = ArticulationCfg(
    spawn=DODO_DAIMAO_SPAWN_CFG,
    init_state=ArticulationCfg.InitialStateCfg(
        # Corrected standing height: feet exactly at z=0 in trajopt
        # Isaac Sim spawns slightly higher to avoid ground penetration
        pos=(0.0, 0.0, 0.55),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Joint order matches CSV_EXPORT_ORDER from backflip script
            "hip_left":       0.0,
            "upper_leg_left": 0.4,
            "lower_leg_left": -0.7,
            "foot_left":      0.3,
            "hip_right":      0.0,
            "upper_leg_right": 0.4,
            "lower_leg_right": -0.7,
            "foot_right":     0.3,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_left", "upper_leg_left", "lower_leg_left", "foot_left",
                "hip_right", "upper_leg_right", "lower_leg_right", "foot_right",
            ],
            effort_limit_sim=30.0,
            velocity_limit_sim=30.0,
            stiffness=80.0,
            damping=0.2,
        ),
    },
)