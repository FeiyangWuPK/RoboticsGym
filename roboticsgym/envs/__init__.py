from gymnasium.envs.registration import register
import gymnasium as gym
from gymnasium.utils.env_checker import check_env


register(
    id="CassieMirror-v0",
    entry_point="roboticsgym.envs.oldcassie_v0:OldCassieMirrorEnv",
    max_episode_steps=600,
)

register(
    id="CassieMirror-v4",
    entry_point="roboticsgym.envs.oldcassie_v4:OldCassieMirrorEnv",
    max_episode_steps=600,
)

register(
    id="CassieMirror-v5",
    entry_point="roboticsgym.envs.oldcassie_v5:OldCassieMirrorEnv",
    max_episode_steps=600,
)

register(
    id="NoisyMujoco-v4",
    entry_point="roboticsgym.envs.noisy_mujoco:NoisyMujocoEnv",
    max_episode_steps=1000,
)
