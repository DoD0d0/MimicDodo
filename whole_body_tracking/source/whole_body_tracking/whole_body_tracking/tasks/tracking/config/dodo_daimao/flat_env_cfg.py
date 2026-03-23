from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from whole_body_tracking.robots.dodo_daimao import DODO_DAIMAO_ROBOT_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from .agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE

# =============================================================================
# USER SETTINGS
# =============================================================================

CONST_AS = 0.25
DODO_DAIMAO_ACTION_SCALE = {
    "hip_left":        CONST_AS,
    "upper_leg_left":  CONST_AS,
    "lower_leg_left":  CONST_AS,
    "foot_left":       CONST_AS,
    "hip_right":       CONST_AS,
    "upper_leg_right": CONST_AS,
    "lower_leg_right": CONST_AS,
    "foot_right":      CONST_AS,
}

# =============================================================================

@configclass
class DodoDaimaoFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = DODO_DAIMAO_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = False

        # Contact sensor — narrow to actual foot links only
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(foot_left|foot_right)",
            history_length=3,
            track_air_time=True,
        )

        # Motion tracking — fill in MISSING fields from base class
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "hip_left",       "upper_leg_left",  "lower_leg_left",  "foot_left",
            "hip_right",      "upper_leg_right", "lower_leg_right", "foot_right",
        ]

        # Termination — fill in MISSING foot link names
        self.terminations.ee_body_pos.params["body_names"] = [
            "foot_left",
            "foot_right",
        ]

        # Actions
        self.actions.joint_pos.scale = DODO_DAIMAO_ACTION_SCALE


# -----------------------------------------------------------------------------
# VARIANTS
# -----------------------------------------------------------------------------

@configclass
class DodoDaimaoFlatWoStateEstimationEnvCfg(DodoDaimaoFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class DodoDaimaoFlatLowFreqEnvCfg(DodoDaimaoFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE