import gymnasium
from gymnasium import spaces
from gymnasium.spaces.dict import Dict
import numpy as np
from typing import Optional
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.envs.mujoco.ant import AntEnv
from gymnasium.envs.mujoco.half_cheetah import HalfCheetahEnv
from gymnasium.envs.mujoco.hopper import HopperEnv
from gymnasium.envs.mujoco.humanoid import HumanoidEnv
from gymnasium.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gymnasium.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gymnasium.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gymnasium.envs.mujoco.pusher import PusherEnv
from gymnasium.envs.mujoco.reacher import ReacherEnv
from gymnasium.envs.mujoco.swimmer import SwimmerEnv
from gymnasium.envs.mujoco.walker2d import Walker2dEnv


class NoisyMujocoEnv(MujocoEnv):
    def __init__(self, task="Ant-v4", domain_randomization_scale=0.0):
        self.domain_randomization_scale = domain_randomization_scale
        self.env = gymnasium.make(task)

        # Define action space
        self.action_space = self.env.action_space
        self.observation_space = spaces.Dict(
            {
                "state": self.env.observation_space,
                "observation": self.env.observation_space,
            }
        )

    def apply_randomization(self, obs: np.ndarray) -> np.ndarray:
        if self.domain_randomization_scale == 0:
            return obs
        else:
            return obs.copy() + np.random.normal(
                scale=self.domain_randomization_scale * np.abs(obs), size=obs.shape
            )

    def _get_obs(self):
        # Get the observation from the environment
        state = self.env._get_obs()
        observation = self.apply_randomization(state)
        ret = {"state": state, "observation": observation}
        return ret

    def set_domain_randomization_scale(self, domain_randomization_scale):
        # Rescale noise level to be between 0 and 1
        self.domain_randomization_scale = domain_randomization_scale

    def reset_model(self):
        # Reset the environment
        state = self.env.reset_model()
        obs = self.apply_randomization(state)
        return {"state": state, "observation": obs}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        state, info = self.env.reset(seed=seed, options=options)
        obs = self.apply_randomization(state)
        return {"state": state, "observation": obs}, info

    def step(self, action):
        observation, reward, terminated, flag, info = self.env.step(action)
        return (
            {
                "state": observation,
                "observation": self.apply_randomization(observation),
            },
            reward,
            terminated,
            flag,
            info,
        )

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
