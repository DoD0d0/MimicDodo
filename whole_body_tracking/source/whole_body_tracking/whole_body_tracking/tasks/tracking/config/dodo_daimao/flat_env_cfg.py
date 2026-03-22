from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg

from whole_body_tracking.robots.dodo_daimao import DODO_DAIMAO_ROBOT_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from .agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE


# =============================================================================
# USER SETTINGS
# =============================================================================

WEIGHT_TRACKING_POS  = 1.0
WEIGHT_TRACKING_ORI  = 1.0
WEIGHT_FEET_AIR_TIME = 0

WEIGHT_ACTION_RATE  = 0.0
WEIGHT_JOINT_ACC    = 0.0
WEIGHT_JOINT_TORQUE = 0.0

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

        # ---------------------------------------------------------------------
        # 1. ROBOT & PHYSICS
        # ---------------------------------------------------------------------
        self.decimation = 2
        self.scene.robot = DODO_DAIMAO_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = False

        # ---------------------------------------------------------------------
        # 2. SENSORS
        # Contact sensors on the foot links (terminal revolute joints)
        # ---------------------------------------------------------------------
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(foot_left|foot_right)",
            history_length=3,
            track_air_time=True,
        )

        # ---------------------------------------------------------------------
        # 3. REWARDS
        # ---------------------------------------------------------------------
        self.rewards.undesired_contacts = None

        # ---------------------------------------------------------------------
        # 4. COMPATIBILITY FIXES
        # dodo_daimao root body is "body" (child of base_link fixed joint)
        # ---------------------------------------------------------------------
        for attr_name in ["events", "rewards"]:
            if hasattr(self, attr_name):
                container = getattr(self, attr_name)
                for name, cfg in container.__dict__.items():
                    if hasattr(cfg, "params") and "asset_cfg" in cfg.params:
                        ac = cfg.params["asset_cfg"]
                        if hasattr(ac, "body_names"):
                            if isinstance(ac.body_names, str) and "torso" in ac.body_names:
                                ac.body_names = "base_link"
                            elif isinstance(ac.body_names, list) and "torso_link" in ac.body_names:
                                ac.body_names = ["base_link"]

        # ---------------------------------------------------------------------
        # 5. ACTIONS & MOTION TRACKING
        # ---------------------------------------------------------------------
        self.actions.joint_pos.scale = DODO_DAIMAO_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base_link"

        # Body names must match link names as Isaac Lab sees them in the URDF.
        # dodo_daimao links: base_link -> body -> hip_* -> upper_leg_* ->
        #                    lower_leg_* -> foot_* -> foot_sole_*
        self.commands.motion.body_names = [
            "base_link",
            "hip_left",       "upper_leg_left",  "lower_leg_left",  "foot_left",
            "hip_right",      "upper_leg_right", "lower_leg_right", "foot_right",
        ]


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