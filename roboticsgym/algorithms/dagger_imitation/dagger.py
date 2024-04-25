"""DAgger (https://arxiv.org/pdf/1011.0686.pdf).

Interactively trains policy by collecting some demonstrations, doing BC, collecting more
demonstrations, doing BC again, etc. Initially the demonstrations just come from the
expert's policy; over time, they shift to be drawn more and more from the imitator's
policy.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Mapping
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit

import numpy as np
import torch as th
from stable_baselines3.common import policies, utils, vec_env
import stable_baselines3.common.logger as sb_logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common import base_class

import torch
from torch.utils.data import DataLoader


from .beta_schedule import BetaSchedule,LinearBetaSchedule
from .trajectory_collector import InteractiveTrajectoryCollector
from .bc import BC
from .rollout import generate_trajectories
from .replay_buffer import ReplayBuffer
from .eval import EvalStudentCallback


DEFAULT_N_EPOCHS: int = 4


class DAggerTrainer(BC):
    """The default number of BC training epochs in `extend_and_update`."""
    def __init__(
        self,
        *,       
        env: Union[GymEnv, str, None],
        policy: Union[str, Type[BasePolicy]] = None,
        learning_rate: Union[float, Schedule] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[torch.device, str] = "auto",
        rng: np.random.Generator,
        batch_size: int = 32,
        n_epochs: int = DEFAULT_N_EPOCHS,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,

        expert_policy: policies.BasePolicy,
        beta_schedule: BetaSchedule = None,
        is_env_noisy: bool = False,
    ):
        """Builds DAggerTrainer.

        Args:
            env: Gym training environment.
            expert_policy: The expert policy used to generate synthetic demonstrations.
            rng: random state for random number generation.
            beta_schedule: Provides a value of `beta` (the probability of taking
                expert action in any given state) at each round of training. If
                `None`, then `linear_beta_schedule` will be used instead.
            bc_trainer: A `BC` instance used to train the underlying policy.

        """
        super().__init__(env=env,
                        policy=policy, 
                        learning_rate=learning_rate, 
                        policy_kwargs=policy_kwargs,
                        device=device,
                        rng=rng,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        optimizer_cls=optimizer_cls,
                        optimizer_kwargs=optimizer_kwargs,
                        ent_weight=ent_weight,
                        l2_weight=l2_weight)

        if beta_schedule is None:
            beta_schedule = LinearBetaSchedule(15)

        self.beta_schedule = beta_schedule
        self.round_num = 0
        self._last_loaded_round = -1
        self.is_env_noisy = is_env_noisy

        self.expert_policy = expert_policy
        # if expert_policy.observation_space != self.venv.observation_space:
        #     raise ValueError("Mismatched observation space between expert_policy and venv")
        
        # if expert_policy.action_space != self.venv.action_space:
        #     raise ValueError("Mismatched action space between expert_policy and venv")
      

    def create_trajectory_collector(self) -> InteractiveTrajectoryCollector:
        """Create trajectory collector to extend current round's demonstration set.

        Returns:
            A collector configured with the appropriate beta, imitator policy, etc.
            for the current round. Refer to the documentation for
            `InteractiveTrajectoryCollector` to see how to use this.
        """
        beta = self.beta_schedule(self.round_num)
        collector = InteractiveTrajectoryCollector(
            env=self.env,
            student_policy=self.policy,
            beta=beta,
            rng=self.rng,
        )

        return collector
    
    def train(
        self,
        *,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        rollout_round_min_episodes: int = 3,
        rollout_round_min_timesteps: int = 500,
        bc_train_kwargs: dict = None,
    ) -> None:
        """Train the DAgger agent.

        The agent is trained in "rounds" where each round consists of a dataset
        aggregation step followed by BC update step.

        During a dataset aggregation step, `self.expert_policy` is used to perform
        rollouts in the environment but there is a `1 - beta` chance (beta is
        determined from the round number and `self.beta_schedule`) that the DAgger
        agent's action is used instead. Regardless of whether the DAgger agent's action
        is used during the rollout, the expert action and corresponding observation are
        always appended to the dataset. The number of environment steps in the
        dataset aggregation stage is determined by the `rollout_round_min*` arguments.

        During a BC update step, `BC.train()` is called to update the DAgger agent on
        all data collected so far.

        Args:
            total_timesteps: The number of timesteps to train inside the environment.
                In practice this is a lower bound, because the number of timesteps is
                rounded up to finish the minimum number of episodes or timesteps in the
                last DAgger training round, and the environment timesteps are executed
                in multiples of `self.venv.num_envs`.
            rollout_round_min_episodes: The number of episodes the must be completed
                completed before a dataset aggregation step ends.
            rollout_round_min_timesteps: The number of environment timesteps that must
                be completed before a dataset aggregation step ends. Also, that any
                round will always train for at least `self.batch_size` timesteps,
                because otherwise BC could fail to receive any batches.
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.
        """

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        total_timestep_count = 0
        self.round_num = 0

        combined_trajectories = []

        callback.on_training_start(locals(), globals())

        while total_timestep_count < total_timesteps:
            print("round: ", self.round_num)
            print("total_timestep_count: ", total_timestep_count)
            collector = self.create_trajectory_collector()

            round_episode_count = 0
            round_timestep_count = 0

            trajectories = generate_trajectories(
                policy=self.expert_policy,
                env=collector,
                callback=callback,
                num_timesteps=total_timestep_count,
            )

            for traj in trajectories:
                combined_trajectories.extend(traj)
                round_timestep_count += len(traj)
                total_timestep_count += len(traj)
                self.num_timesteps = total_timestep_count

            round_episode_count += len(trajectories)


            rb = ReplayBuffer(combined_trajectories)

            data_loader = DataLoader(rb, batch_size=self.batch_size,
                                           shuffle=True, num_workers=4, pin_memory=True)
            
            self.set_demonstrations(data_loader)

            super().learn()

            self.round_num += 1
            return self.round_num
        
        callback.on_training_end()
