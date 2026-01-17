import gymnasium as gym

# <--- KORRIGIERT: Importiere deine neuen Dodo-Klassen
from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-Dodo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DodoFlatEnvCfg, 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DodoFlatPPORunnerCfg", 
    },
)

gym.register(
    id="Tracking-Flat-Dodo-Wo-State-Estimation-v0", 
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DodoFlatWoStateEstimationEnvCfg, 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DodoFlatPPORunnerCfg", 
    },
)


gym.register(
    id="Tracking-Flat-Dodo-Low-Freq-v0", 
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DodoFlatLowFreqEnvCfg, 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DodoFlatLowFreqPPORunnerCfg", 
    },
)