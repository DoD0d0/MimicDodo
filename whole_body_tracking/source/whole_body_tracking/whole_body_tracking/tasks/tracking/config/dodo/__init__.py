import gymnasium as gym

# <--- KORRIGIERT: Importiere deine neuen Dodo-Klassen
from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-Dodo-v0", # <--- KORRIGIERT: Neuer Task-Name
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DodoFlatEnvCfg, # <--- KORRIGIERT
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DodoFlatPPORunnerCfg", # <--- KORRIGIERT
    },
)

gym.register(
    id="Tracking-Flat-Dodo-Wo-State-Estimation-v0", # <--- KORRIGIERT
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DodoFlatWoStateEstimationEnvCfg, # <--- KORRIGIERT
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DodoFlatPPORunnerCfg", # <--- KORRIGIERT
    },
)


gym.register(
    id="Tracking-Flat-Dodo-Low-Freq-v0", # <--- KORRIGIERT
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DodoFlatLowFreqEnvCfg, # <--- KORRIGIERT
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DodoFlatLowFreqPPORunnerCfg", # <--- KORRIGIERT
    },
)