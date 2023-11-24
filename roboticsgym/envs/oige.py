# This file is based on torchrl's isaacgym.py,
# which is intended for isaac gym preview.
from __future__ import annotations

import importlib.util

import itertools
import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import gymnasium as gym

from tensordict import TensorDictBase
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import _classproperty, make_composite_from_td

_has_isaac = importlib.util.find_spec("omniisaacgymenvs") is not None


class OIGEWrapper(GymWrapper):
    """Wrapper for OmniIsaacGymEnvs environments.

    The original library can be found `here <https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs>`_
    and is based on Issac Sim Gym Extension, which can be downloaded `through NVIDIA's webpage <https://developer.nvidia.com>_`.

    .. note:: IsaacGym environments cannot be executed consecutively, ie. instantiating one
        environment after another (even if it has been cleared) will cause
        CUDA memory issues. We recommend creating one environment per process only.
        If you need more than one environment, the best way to achieve that is
        to spawn them across processes.

    .. note:: IsaacGym works on CUDA devices by essence. Make sure your machine
        has GPUs available and the required setup for IsaacGym.

    """

    @property
    def lib(self):
        import omniisaacgymenvs

        return omniisaacgymenvs

    def __init__(
        self, env: "omniisaacgymenvs.envs.vec_envs.VecEnv", **kwargs
    ):  # noqa: F821
        warnings.warn(
            "OmniIsaacGym environment support is an experimental feature that may change in the future."
        )
        num_envs = env.num_envs
        super().__init__(
            env, torch.device(env.device), batch_size=torch.Size([num_envs]), **kwargs
        )
        if not hasattr(self, "task"):
            # by convention in OmniIsaacGymEnvs
            self.task = env.__name__

    def _make_specs(self, env: "gym.Env") -> None:  # noqa: F821
        super()._make_specs(env, batch_size=self.batch_size)
        self.full_done_spec = {
            key: spec.squeeze(-1) for key, spec in self.full_done_spec.items(True, True)
        }
        self.observation_spec["obs"] = self.observation_spec["observation"]
        del self.observation_spec["observation"]

        data = self.rollout(3).get("next")[..., 0]
        del data[self.reward_key]
        for done_key in self.done_keys:
            try:
                del data[done_key]
            except KeyError:
                continue
        specs = make_composite_from_td(data)

        obs_spec = self.observation_spec
        obs_spec.unlock_()
        obs_spec.update(specs)
        obs_spec.lock_()
        self.__dict__["full_observation_spec"] = obs_spec

    @classmethod
    def _make_envs(cls, *, task, num_envs, device, seed=None, headless=True, **kwargs):
        from omniisaacgymenvs.envs.vec_env import VecEnv
        import omegaconf, os
        from omniisaacgymenvs.utils.hydra_cfg.reformat import (
            omegaconf_to_dict,
            print_dict,
        )
        import datetime
        from omniisaacgymenvs.utils.config_utils.path_utils import (
            retrieve_checkpoint_path,
            get_experience,
        )

        cfg = omegaconf.OmegaConf.load("../configs/config.yaml")
        cls.cfg = cfg
        cfg.task = task
        cfg.num_envs = num_envs
        cfg.sim_device = device
        cfg.headless = headless

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cfg.headless = headless
        # local rank (GPU id) in a current multi-gpu mode
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        # global rank (GPU id) in multi-gpu multi-node mode
        global_rank = int(os.getenv("RANK", "0"))
        cls.global_rank = global_rank
        if cfg.multi_gpu:
            cfg.device_id = local_rank
            cfg.rl_device = f"cuda:{local_rank}"
        enable_viewport = (
            "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
        )

        # select kit app file
        experience = get_experience(
            headless, cfg.enable_livestream, enable_viewport, cfg.kit_app
        )

        envs = VecEnv(
            headless=headless,
            sim_device=cfg.device_id,
            enable_livestream=cfg.enable_livestream,
            enable_viewport=enable_viewport,
            experience=experience,
        )
        cls.envs = envs
        # ensure checkpoints can be specified as relative paths
        if cfg.checkpoint:
            cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
            if cfg.checkpoint is None:
                quit()

        cfg_dict = omegaconf_to_dict(cfg)
        print_dict(cfg_dict)
        # sets seed. if seed is -1 will pick a random one
        from omni.isaac.core.utils.torch.maths import set_seed

        cfg.seed = seed
        cfg.seed = cfg.seed + global_rank if cfg.seed != -1 else cfg.seed
        cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
        cfg_dict["seed"] = cfg.seed
        from omniisaacgymenvs.utils.task_util import initialize_task

        task = initialize_task(cfg_dict, envs)

        if cfg.wandb_activate and global_rank == 0:
            # Make sure to install WandB if you actually use this.
            import wandb

            run_name = f"{cfg.wandb_name}_{time_str}"

            wandb.init(
                project=cfg.wandb_project,
                group=cfg.wandb_group,
                entity=cfg.wandb_entity,
                config=cfg_dict,
                sync_tensorboard=True,
                name=run_name,
                resume="allow",
            )

        torch.cuda.set_device(local_rank)
        return envs

    def _set_seed(self, seed: int) -> int:
        # as of #665c32170d84b4be66722eea405a1e08b6e7f761 the seed points nowhere in gym.make for IsaacGymEnvs
        return seed

    def read_action(self, action):
        """Reads the action obtained from the input TensorDict and transforms it in the format expected by the contained environment.

        Args:
            action (Tensor or TensorDict): an action to be taken in the environment

        Returns: an action in a format compatible with the contained environment.

        """
        return action

    def read_done(
        self,
        terminated: bool = None,
        truncated: bool | None = None,
        done: bool | None = None,
    ) -> Tuple[bool, bool, bool]:
        if terminated is not None:
            terminated = terminated.bool()
        if truncated is not None:
            truncated = truncated.bool()
        if done is not None:
            done = done.bool()
        return terminated, truncated, done, done.any()

    def read_reward(self, total_reward, step_reward):
        """Reads a reward and the total reward so far (in the frame skip loop) and returns a sum of the two.

        Args:
            total_reward (torch.Tensor or TensorDict): total reward so far in the step
            step_reward (reward in the format provided by the inner env): reward of this particular step

        """
        return total_reward + step_reward

    def read_obs(
        self, observations: Union[Dict[str, Any], torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """Reads an observation from the environment and returns an observation compatible with the output TensorDict.

        Args:
            observations (observation under a format dictated by the inner env): observation to be read.

        """
        if isinstance(observations, dict):
            if "state" in observations and "observation" not in observations:
                # we rename "state" in "observation" as "observation" is the conventional name
                # for single observation in torchrl.
                # naming it 'state' will result in envs that have a different name for the state vector
                # when queried with and without pixels
                observations["observation"] = observations.pop("state")
        if not isinstance(observations, (TensorDictBase, dict)):
            (key,) = itertools.islice(self.observation_spec.keys(True, True), 1)
            observations = {key: observations}
        return observations

    def __del__(self):
        self.envs.close()

        if self.cfg.wandb_activate and self.global_rank == 0:
            import wandb

            wandb.finish()
        return super().__del__()


class IsaacGymEnv(OIGEWrapper):
    """A TorchRL Env interface for IsaacGym environments.

    See :class:`~.IsaacGymWrapper` for more information.

    Examples:
        >>> env = IsaacGymEnv(task="Ant", num_envs=2000, device="cuda:0")
        >>> rollout = env.rollout(3)
        >>> assert env.batch_size == (2000,)

    """

    @_classproperty
    def available_envs(cls):
        if not _has_isaac:
            return

        import omniisaacgymenvs  # noqa

        from omniisaacgymenvs.utils.task_util import import_tasks

        task_map, task_map_warp = import_tasks()
        yield from task_map.keys()
        yield from task_map_warp.keys()

    def __init__(self, task=None, *, env=None, num_envs, device, **kwargs):
        if env is not None and task is not None:
            raise RuntimeError("Cannot provide both `task` and `env` arguments.")
        elif env is not None:
            task = env
        envs = self._make_envs(task=task, num_envs=num_envs, device=device, **kwargs)
        self.task = task
        super().__init__(envs, **kwargs)
