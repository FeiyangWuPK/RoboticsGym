"""DAgger (https://arxiv.org/pdf/1011.0686.pdf).

Interactively trains policy by collecting some demonstrations, doing BC, collecting more
demonstrations, doing BC again, etc. Initially the demonstrations just come from the
expert's policy; over time, they shift to be drawn more and more from the imitator's
policy.
"""

import pathlib


import numpy as np
import torch as th
from stable_baselines3.common import policies, utils, vec_env
import stable_baselines3.common.logger as sb_logger
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common import base_class


from torch.utils.data import DataLoader


from .beta_schedule import BetaSchedule,LinearBetaSchedule
from .trajectory_collector import InteractiveTrajectoryCollector
from .bc import BC
from .rollout import generate_trajectories
from .replay_buffer import ReplayBuffer
from .eval import EvalStudentCallback


DEFAULT_N_EPOCHS: int = 4


class DAggerTrainer():
    """The default number of BC training epochs in `extend_and_update`."""
    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        rng: np.random.Generator,
        beta_schedule: BetaSchedule = None,
        bc_trainer: BC,
        is_env_noisy: bool = False,
    ):
        """Builds DAggerTrainer.

        Args:
            venv: Vectorized training environment.
            rng: random state for random number generation.
            beta_schedule: Provides a value of `beta` (the probability of taking
                expert action in any given state) at each round of training. If
                `None`, then `linear_beta_schedule` will be used instead.
            bc_trainer: A `BC` instance used to train the underlying policy.

        """
        super().__init__()

        if beta_schedule is None:
            beta_schedule = LinearBetaSchedule(15)

        self.beta_schedule = beta_schedule
        self.venv = venv
        self.round_num = 0
        self._last_loaded_round = -1
        self._all_demos = []
        self.rng = rng

        # utils.check_for_correct_spaces(
        #     self.venv,
        #     bc_trainer.observation_space,
        #     bc_trainer.action_space,
        # )
        self.bc_trainer = bc_trainer
        self.is_env_noisy = is_env_noisy


    def __getstate__(self):
        """Return state excluding non-pickleable objects."""
        d = dict(self.__dict__)
        del d["venv"]

        return d

    @property
    def policy(self) -> policies.BasePolicy:
        return self.bc_trainer.policy

    @property
    def batch_size(self) -> int:
        return self.bc_trainer.batch_size

    def extend_and_update(
        self,
        bc_train_kwargs: dict = None,
    ) -> int:
        """Extend internal batch of data and train BC.

        Specifically, this method will load new transitions (if necessary), train
        the model for a while, and advance the round counter. If there are no fresh
        demonstrations in the demonstration directory for the current round, then
        this will raise a `NeedsDemosException` instead of training or advancing
        the round counter. In that case, the user should call
        `.create_trajectory_collector()` and use the returned
        `InteractiveTrajectoryCollector` to produce a new set of demonstrations for
        the current interaction round.

        Arguments:
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.

        Returns:
            New round number after advancing the round counter.
        """
        if bc_train_kwargs is None:
            bc_train_kwargs = {}
        else:
            bc_train_kwargs = dict(bc_train_kwargs)

        user_keys = bc_train_kwargs.keys()
        if "log_rollouts_venv" not in user_keys:
            bc_train_kwargs["log_rollouts_venv"] = self.venv

        if "n_epochs" not in user_keys and "n_batches" not in user_keys:
            bc_train_kwargs["n_epochs"] = DEFAULT_N_EPOCHS


        self.bc_trainer.train(**bc_train_kwargs)
        self.round_num += 1
        return self.round_num

    def create_trajectory_collector(self) -> InteractiveTrajectoryCollector:
        """Create trajectory collector to extend current round's demonstration set.

        Returns:
            A collector configured with the appropriate beta, imitator policy, etc.
            for the current round. Refer to the documentation for
            `InteractiveTrajectoryCollector` to see how to use this.
        """

        beta = self.beta_schedule(self.round_num)
        collector = InteractiveTrajectoryCollector(
            venv=self.venv,
            student_policy=self.bc_trainer.policy,
            beta=beta,
            rng=self.rng,
            is_env_noisy=self.is_env_noisy,
        )
        return collector
    

    def save_policy(self,path):
        self.bc_trainer.policy.save(path)


class SimpleDAggerTrainer(DAggerTrainer):
    """Simpler subclass of DAggerTrainer for training with synthetic feedback."""

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        expert_policy: policies.BasePolicy,
        rng: np.random.Generator,
        expert_trajs: list = None,
        **dagger_trainer_kwargs,
    ):
        """Builds SimpleDAggerTrainer.

        Args:
            venv: Vectorized training environment. Note that when the robot
                action is randomly injected (in accordance with `beta_schedule`
                argument), every individual environment will get a robot action
                simultaneously for that timestep.

            expert_policy: The expert policy used to generate synthetic demonstrations.
            rng: Random state to use for the random number generator.
            expert_trajs: Optional starting dataset that is inserted into the round 0
                dataset.
            dagger_trainer_kwargs: Other keyword arguments passed to the
                superclass initializer `DAggerTrainer.__init__`.

        Raises:
            ValueError: The observation or action space does not match between
                `venv` and `expert_policy`.
        """
        super().__init__(
            venv=venv,
            rng=rng,
            **dagger_trainer_kwargs,
        )

        self.expert_policy = expert_policy
        # if expert_policy.observation_space != self.venv.observation_space:
        #     raise ValueError("Mismatched observation space between expert_policy and venv")
        
        # if expert_policy.action_space != self.venv.action_space:
        #     raise ValueError("Mismatched action space between expert_policy and venv")


    def train(
        self,
        *,
        total_timesteps: int,
        callback: EvalStudentCallback,
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
        total_timestep_count = 0
        round_num = 0

        combined_trajectories = []
        self.venv.reset()

        callback.init_callback(self.bc_trainer)

        while total_timestep_count < total_timesteps:
            print("round: ", round_num)
            print("total_timestep_count: ", total_timestep_count)
            collector = self.create_trajectory_collector()
            round_episode_count = 0
            round_timestep_count = 0

            trajectories = generate_trajectories(
                policy=self.expert_policy,
                venv=collector,
                callback=callback,
                is_env_noisy=self.is_env_noisy,
                num_timesteps=total_timestep_count,
            )

            for traj in trajectories:
                combined_trajectories.extend(traj)
                round_timestep_count += len(traj)
                total_timestep_count += len(traj)

            round_episode_count += len(trajectories)

            # self.logger.record("dagger/total_timesteps_count", total_timestep_count)
            # self.logger.record("dagger/round_num", round_num)
            # self.logger.record("dagger/round_episode_count", round_episode_count)
            # self.logger.record("dagger/round_timestep_count", round_timestep_count)

            rb = ReplayBuffer(combined_trajectories)

            data_loader = DataLoader(rb, batch_size=self.batch_size,
                                           shuffle=True, num_workers=4, pin_memory=True)
            
            self.bc_trainer.set_demonstrations(data_loader)

            self.extend_and_update(bc_train_kwargs)
            round_num += 1