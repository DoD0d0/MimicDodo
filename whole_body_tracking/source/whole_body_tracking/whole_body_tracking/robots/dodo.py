# from tkinter.constants import FALSE
import isaaclab.sim as sim_utils
import os
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Calculate path to centralized MimicDodo assets folder
# From: whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/
# To: MimicDodo/assets/dodo/urdf/dodo.urdf
# Going up 5 levels: robots/ -> whole_body_tracking(package) -> whole_body_tracking(source) -> whole_body_tracking(project) -> MimicDodo/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "../../../../../"))
_ASSET_DIR = os.path.join(_project_root, "assets/dodo")


DODO_SPAWN_CFG = sim_utils.UrdfFileCfg(
    asset_path=os.path.join(_ASSET_DIR, "urdf/dodo.urdf"),  # Using centralized dodo.urdf from MimicDodo/assets
    activate_contact_sensors=True,  
    fix_base=False,  # Changed from fix_root_link in articulation_props
    replace_cylinders_with_capsules=False,

    # 🔴 FORCE NEW USD GENERATION 🔴
    # This ensures your new motor signs are compiled!
    force_usd_conversion=True,

    # This tells the importer how to configure the joints in the generated USD
    joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
        drive_type="force",  # We usually use "force" for torque-controlled robots
        target_type="position",
        gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,  # Initialize to 0.0 (we control it via Actuators later)
            damping=0.0,    # Initialize to 0.0
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
        # fix_root_link is now fix_base in UrdfFileCfg (moved up)
    ),
)


DODO_ROBOT_CFG = ArticulationCfg(
    spawn=DODO_SPAWN_CFG,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6), # Perfect standing height # 0.53
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_joint_1": 0.0, 
            "left_joint_2": 0.4,    #  0.436, 
            "left_joint_3": -0.7,   # -0.611, 
            "left_joint_4": 0.3,    #  0.175,
            "right_joint_1": 0.0, 
            "right_joint_2": 0.4,   #  0.436, 
            "right_joint_3": -0.7,  # -0.611, 
            "right_joint_4": 0.3,   #  0.175,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            # Select ALL joints
            joint_names_expr=[
            'left_joint_1', 'right_joint_1', 
            'left_joint_2', 'right_joint_2', 
            'left_joint_3', 'right_joint_3', 
            'left_joint_4', 'right_joint_4'
            ],
            effort_limit_sim=30.0,              # was 6 Newton Meters for walking  / 30 for Backflip => robot actually only needs 7Nm
            velocity_limit_sim=30.0,            # was 15 rad/s for walking         / 30 for Backflip => robot actually only needs 5Nm
            stiffness=80.0,                     # KP:  80 for walking                            
            damping=0.2                         # KD:  1 for walking    # TODO test 0.2 for Backflip
        ),
    },
)