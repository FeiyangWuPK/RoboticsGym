from gymnasium.envs.registration import register
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

register(
    id="CassieViz-v1",
    entry_point="roboticsgym.envs.cassie_viz:CassieEnv",
    max_episode_steps=1000,
)

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
    id="CassieMirror-v6",
    entry_point="roboticsgym.envs.oldcassie_v6:OldCassieMirrorEnv",
    max_episode_steps=600,
)

register(
    id="NoisyMujoco-v4",
    entry_point="roboticsgym.envs.noisy_mujoco:NoisyMujocoEnv",
    max_episode_steps=1000,
)

register(
    id="Digit-v1",
    entry_point="roboticsgym.envs.digit:DigitEnv",
    max_episode_steps=2000,
)

register(
    id="Digit-v2",
    entry_point="roboticsgym.envs.digit_v2:DigitEnv",
    max_episode_steps=2000,
)

register(
    id="DigitViz-v1",
    entry_point="roboticsgym.envs.digit_viz:DigitEnv",
    max_episode_steps=2000,
)

register(
    id="DigitFKHY-v1",
    entry_point="roboticsgym.envs.digit_fkhy:DigitEnvFlat",
    max_episode_steps=800,
)
