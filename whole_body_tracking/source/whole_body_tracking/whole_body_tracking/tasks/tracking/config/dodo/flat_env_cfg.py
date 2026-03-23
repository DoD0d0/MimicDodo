from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from whole_body_tracking.robots.dodo import DODO_ROBOT_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from .agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE

# =============================================================================
# USER SETTINGS
# =============================================================================

CONST_AS = 0.25
DODO_ACTION_SCALE = {
    "left_joint_1": CONST_AS,
    "left_joint_2": CONST_AS,
    "left_joint_3": CONST_AS,
    "left_joint_4": CONST_AS,
    "right_joint_1": CONST_AS,
    "right_joint_2": CONST_AS,
    "right_joint_3": CONST_AS,
    "right_joint_4": CONST_AS,
}

# =============================================================================

@configclass
class DodoFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = DODO_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = False

        # Contact sensor — narrow to actual foot links only
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(left|right)_link_4",
            history_length=3,
            track_air_time=True,
        )

        # Motion tracking — fill in MISSING fields from base class
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "left_link_1",  "left_link_2",  "left_link_3",  "left_link_4",
            "right_link_1", "right_link_2", "right_link_3", "right_link_4",
        ]

        # Termination — fill in MISSING foot link names
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_link_4",
            "right_link_4",
        ]

        # Actions
        self.actions.joint_pos.scale = DODO_ACTION_SCALE


# -----------------------------------------------------------------------------
# VARIANTS
# -----------------------------------------------------------------------------

@configclass
class DodoFlatWoStateEstimationEnvCfg(DodoFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class DodoFlatLowFreqEnvCfg(DodoFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE