from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg 
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.envs.mdp import root_height_below_minimum

from whole_body_tracking.robots.dodo import DODO_ROBOT_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from .agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE

# =============================================================================
# 🎛️  USER SETTINGS (TWEAK WEIGHTS HERE)
# =============================================================================

# 1. THE MAIN TRADEOFF (Ghost vs. Feet)
WEIGHT_TRACKING_POS   = 1.0   # Lowered (was 8.0) -> Allows small deviations to lift legs
WEIGHT_TRACKING_ORI   = 1.0   # Lowered (was 8.0)
WEIGHT_FEET_AIR_TIME  = 0   

# 2. PENALTIES (Keep the robot smooth)
WEIGHT_ACTION_RATE    = 0.0        # Original: -0.1 // Also tested: -0.005, -0.02 on 2026-01-12_14-59-58
WEIGHT_JOINT_ACC      = 0.0
WEIGHT_JOINT_TORQUE   = 0.0

# 3. ACTION SCALES
CONST_AS = 0.25 # 0.1
DODO_ACTION_SCALE = {
    "left_joint_1": CONST_AS, "left_joint_2": CONST_AS, "left_joint_3": CONST_AS, "left_joint_4": CONST_AS,
    "right_joint_1": CONST_AS, "right_joint_2": CONST_AS, "right_joint_3": CONST_AS, "right_joint_4": CONST_AS,
}

# =============================================================================

@configclass
class DodoFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------------
        # 1. ROBOT & PHYSICS SETUP
        # ---------------------------------------------------------------------
        self.decimation = 2  # <--- WAS 4. Change to 2!
        
        # Math:
        # Physics dt (0.005) * Decimation (2) = 0.01s Control Step
        # 1 / 0.01s = 100 Hz Control Frequency

        
        # Load the robot config
        self.scene.robot = DODO_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 1. DISABLE THE ANCHOR ("root_joint")
        # This tells Isaac Lab: "Don't create a fixed joint to the world."
        # This resolves the "CreateJoint - found a joint with disjointed body transforms" error.
        self.scene.robot.spawn.articulation_props.fix_root_link = False

        # ---------------------------------------------------------------------
        # 2. SENSORS (The "Ghost" Fix)
        # ---------------------------------------------------------------------
        # Targets 'left_link_4' or 'right_link_4' inside 'dodo' robot.
        # The '$' ensures we don't accidentally grab the visual mesh child.
        self.scene.contact_forces = ContactSensorCfg(
            # prim_path="{ENV_REGEX_NS}/Robot/dodo/(left|right)_link_4$",        # for USD
            prim_path="{ENV_REGEX_NS}/Robot/(left|right)_link_4",                # for URDF
            # This matches ".../Robot/link_4" for the dodo robot
            # prim_path="{ENV_REGEX_NS}/Robot/.*/(left|right)_link_4",
            history_length=3, 
            track_air_time=True,
        )

        # ---------------------------------------------------------------------
        # 3. REWARDS
        # ---------------------------------------------------------------------

        # Disable automatic contact penalties (we handle this manually)
        self.rewards.undesired_contacts = None 

        # ---------------------------------------------------------------------
        # 4. COMPATIBILITY FIXES (The "Torso" Loops)
        # ---------------------------------------------------------------------
        # Renames "torso" to "base_link" in all events/rewards if needed
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
        # 5. TERMINATION & ACTIONS
        # ---------------------------------------------------------------------
        
        # --- Deactivate terminations for testing ---
        # self.terminations.anchor_ori = None
        # self.terminations.anchor_pos = None
        # self.terminations.ee_body_pos = None
        # self.terminations.time_out = None
        # -------------------------------------------

        # self.commands.motion.resampling_time_range = (0.0, 0.0) 
        self.actions.joint_pos.scale = DODO_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "left_link_1", "left_link_2", "left_link_3", "left_link_4",
            "right_link_1", "right_link_2", "right_link_3", "right_link_4",
        ]
        # breakpoint()

        

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