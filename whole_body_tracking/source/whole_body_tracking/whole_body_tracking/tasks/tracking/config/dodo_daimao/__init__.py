import gymnasium as gym

from . import agents, flat_env_cfg

gym.register(
    id="Tracking-Flat-DodoDaimao-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DodoDaimaoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DodoDaimaoFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DodoDaimao-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DodoDaimaoFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DodoDaimaoFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DodoDaimao-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DodoDaimaoFlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DodoDaimaoFlatLowFreqPPORunnerCfg",
    },
)