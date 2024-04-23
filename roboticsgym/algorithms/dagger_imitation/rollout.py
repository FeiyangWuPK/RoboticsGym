
import numpy as np

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv
from .eval import EvalStudentCallback

from .trajectory_accumulator import TrajectoryAccumulator
    
def generate_trajectories(
    policy: BasePolicy,
    venv: VecEnv,
    callback: EvalStudentCallback,
    is_env_noisy: bool = False,
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
    
    num_actions = venv.action_space.shape[0]
    if is_env_noisy:
        num_obs = venv.observation_space['observation'].shape[0]
    else:
        num_obs = venv.observation_space.shape[0]

    trajectories_accum = TrajectoryAccumulator(venv.num_envs, num_obs, num_actions)

    obs = venv.reset()

    active = np.ones(venv.num_envs, dtype=bool)
    dones = np.zeros(venv.num_envs, dtype=bool)

    print("Dagger Generate Trajectories")
    while np.any(active):
        acts, _ = policy.predict(obs,deterministic=True)
        next_obs, rews, dones, _ = venv.step(acts)

        callback.on_step(num_timesteps=num_timesteps)

        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            obs[active],
            acts[active],
            rews[active],
            dones[active]
        )

        obs = next_obs

        trajectories.extend(new_trajs)

        active &= ~dones

    return trajectories
