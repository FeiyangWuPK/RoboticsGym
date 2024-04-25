
import numpy as np
from typing import Dict

import gymnasium as gym

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.callbacks import BaseCallback


from .trajectory_accumulator import TrajectoryAccumulator
    
def generate_trajectories(
    expert_policy: BasePolicy,
    env: GymEnv,
    callback: BaseCallback,
    num_timesteps: int = 0,
      ):
    """Generate trajectory dictionaries from a policy and an environment.
    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments to interact with.
        sample_until: A function determining the termination condition.
            It takes a sequence of trajectories, and returns a bool.
            Most users will want to use one of `min_episodes` or `min_timesteps`.
        deterministic_policy: If True, asks policy to deterministically return
            action. Note the trajectories might still be non-deterministic if the
            environment has non-determinism!

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    trajectories = []

    obs = env.reset()

    if isinstance(obs, Dict):
        state = obs['state']
        obs = obs['observation']
    else:
        state = obs
            
    trajectories_accum = TrajectoryAccumulator(env.num_envs)

    active = np.ones(env.num_envs, dtype=bool)
    dones = np.zeros(env.num_envs, dtype=bool)

    print("Generate Trajectories")
    callback.on_rollout_start()

    while np.any(active):
        acts, _ = expert_policy.predict(state, deterministic=True)
        next_obs, rews, dones, info = env.step(acts)

        callback.update_locals(locals())
        callback.on_step()

        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            state[active],
            obs[active],
            acts[active],
            rews[active],
            dones[active]
        )

        if isinstance(next_obs, Dict):
            state = next_obs['state']
            obs = next_obs['observation']
        else:
            state = next_obs         

        trajectories.extend(new_trajs)

        active &= ~dones

    callback.on_rollout_end()
    return trajectories
