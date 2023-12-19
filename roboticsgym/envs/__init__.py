from gymnasium.envs.registration import register
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

register(
    id="CassieMirror-v1",
    entry_point="roboticsgym.envs.oldcassie:OldCassieMirrorEnv",
    max_episode_steps=600,
)

register(
    id="CassieMirror-v4",
    entry_point="roboticsgym.envs.oldcassie_v4:OldCassieMirrorEnv",
    max_episode_steps=600,
)
