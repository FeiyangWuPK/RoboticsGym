
import numpy as np
from typing import Dict

import gymnasium as gym

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.policies import BasePolicy

class InteractiveTrajectoryCollector(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, 
                 env: GymEnv,
                 student_policy: BasePolicy,
                 beta: float,
                 rng: np.random.Generator,
                 ):
        super().__init__(env)        
        assert 0 <= beta <= 1
        self.beta = beta
        self._last_obs = None
        self._done_before = True
        self._is_reset = False
        self._last_user_actions = None
        self.rng = rng

        self.student_policy = student_policy

        
    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()

        self._last_obs = obs
        self._is_reset = True
        self._last_user_actions = None

        return obs
    

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is this a final state (episode finished),
        is the max number of steps reached (episode finished artificially), additional informations
        """
        assert self._is_reset, "call .reset() before .step()"
        assert self._last_obs is not None
        assert action is not None

        actual_acts = np.array(action)

        if isinstance(self._last_obs, Dict):
            obs = self._last_obs['observation']

        mask = self.rng.uniform(0, 1, size=(self.num_envs,)) > self.beta
        if np.sum(mask) != 0:
            actual_acts[mask],_ = self.student_policy.predict(obs[mask])

        next_obs, reward, done, info = self.env.step(actual_acts)

        self._last_user_actions = action
        self._last_obs = next_obs

        return next_obs, reward, done, info
    
