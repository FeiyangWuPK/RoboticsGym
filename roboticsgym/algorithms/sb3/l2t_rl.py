import os
import io
import warnings
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Callable

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import (
    ReplayBuffer,
    DictReplayBuffer,
)
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    get_parameters_by_name,
    polyak_update,
    get_schedule_fn,
)
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common import type_aliases
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    sync_envs_normalization,
    is_vecenv_wrapped,
)

from roboticsgym.algorithms.sb3.policies import (
    Actor,
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
    L2TPolicy,
)


try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

SelfL2TRL = TypeVar("SelfL2TRL", bound="L2TRL")


class L2TRL(OffPolicyAlgorithm):
    """
    Hallucinated Inverse Policy mirror descent algorithm (L2T)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: TODO
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    Additionally, we include a teacher actor and student actor.
    The teacher actor is trained to mimic the reference trajectories using Imitation Learning,
    while the student actor is trained to mimic the teacher actor.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param expert_replaybuffer: Path to the expert replay buffer
    :param expert_replaybuffersize: Size of the expert replay buffer
    :param student_begin: Begin training student agent after this many timesteps
    :param reward_reg_param: Reward regularization parameter
    :param teacher_state_only_reward: Teacher state only reward, note that although action can still be passed as arguments, but it does not participate the reward funciton computation, i.e., r(s,a)=r(s) if teacher_state_only_reward=True.
    :param student_domain_randomization_scale: Student domain randomization scale
    :param explorer: Teacher or student
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "L2TPolicy": L2TPolicy,
    }
    policy: L2TPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[L2TPolicy]],
        env: Union[GymEnv, str],
        student_policy: Union[str, Type[L2TPolicy]] = L2TPolicy,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        student_gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        student_ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        clip_range: Union[float, Schedule] = 0.2,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        student_policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        student_domain_randomization_scale: float = 0.0,
        explorer: str = "teacher",
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.student_gamma = student_gamma
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.student_ent_coef = student_ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.student_ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.student_policy = student_policy
        self.student_policy_kwargs = (
            {} if student_policy_kwargs is None else student_policy_kwargs
        )

        # Update policy keyword arguments
        self.policy_kwargs["use_sde"] = self.use_sde
        self.student_policy_kwargs["use_sde"] = self.use_sde

        self.clip_range = clip_range
        self.explorer = explorer
        self.student_domain_randomization_scale = student_domain_randomization_scale

        if _init_setup_model:
            self._setup_model()

    def _init_student_policy(
        self,
        student_policy: Union[str, Type[L2TPolicy]] = L2TPolicy,
        student_policy_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(student_policy, str):
            self.student_policy_class = self._get_policy_from_name(student_policy)
        else:
            self.student_policy_class = student_policy

        self.student_policy_kwargs = student_policy_kwargs

        # from off_policy_algorithm super()._setup_model()
        # partial_obversevation_space is from Environment's partial_observation_space
        self.partial_observation_space = self.observation_space["observation"]  # type: ignore
        self.student_policy = self.student_policy_class(  # pytype:disable=not-instantiable
            self.partial_observation_space,
            self.action_space,
            self.lr_schedule,
            **self.student_policy_kwargs,  # pytype:disable=not-instantiable # type: ignore
        )
        self.student_policy = self.student_policy.to(self.device)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            if issubclass(self.replay_buffer_class, HerReplayBuffer):
                assert (
                    self.env is not None
                ), "You must pass an environment when using `HerReplayBuffer`"
                replay_buffer_kwargs["env"] = self.env
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,  # type: ignore
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,
            )

        self.policy = self.policy_class(  # type: ignore
            self.observation_space["state"],  # type: ignore
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)  # type: ignore

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

        self.clip_range = get_schedule_fn(self.clip_range)
        self._init_student_policy(self.student_policy, self.student_policy_kwargs)  # type: ignore
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert (
                    init_value > 0.0
                ), "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

        if isinstance(self.student_ent_coef, str) and self.student_ent_coef.startswith(
            "auto"
        ):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.student_ent_coef:
                init_value = float(self.student_ent_coef.split("_")[1])
                assert (
                    init_value > 0.0
                ), "The initial value of student_ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_student_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.student_ent_coef_optimizer = th.optim.Adam(
                [self.log_student_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.student_ent_coef_tensor = th.tensor(
                float(self.student_ent_coef), device=self.device
            )

        self.student_batch_norm_stats = get_parameters_by_name(
            self.student_critic, ["running_"]
        )
        self.student_batch_norm_stats_target = get_parameters_by_name(
            self.student_critic_target, ["running_"]
        )

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        self.student_actor = self.student_policy.actor  # type: ignore
        self.student_critic = self.student_policy.critic  # type: ignore
        self.student_critic_target = self.student_policy.critic_target  # type: ignore

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.student_policy.set_training_mode(True)  # type: ignore
        # Update optimizers learning rate
        optimizers = [
            self.actor.optimizer,
            self.critic.optimizer,
            self.student_actor.optimizer,
            self.student_critic.optimizer,
        ]  # reward est optimizer should not change pace
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        if self.student_ent_coef_optimizer is not None:
            optimizers += [self.student_ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        # update_learning_rate(self.student_reward_est.optimizer, self.lr_schedule(self._current_progress_remaining) * 5)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        # For student agent
        student_actor_losses, student_critic_losses = [], []
        student_ent_coef_losses, student_ent_coefs = [], []
        average_reward_list = []

        student_average_reward_list = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()
                self.student_actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations["state"]  # type: ignore
            )
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            average_reward_list.append(replay_data.rewards.mean())
            # teacher update critic and actor
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations["state"]  # type: ignore
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(
                        replay_data.next_observations["state"], next_actions  # type: ignore
                    ),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - (
                    self.gamma != 1
                ) * ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

                target_q_values -= (self.gamma == 1) * replay_data.rewards.mean()

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(
                replay_data.observations["state"], replay_data.actions  # type: ignore
            )

            # Compute critic loss
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(
                self.critic(replay_data.observations["state"], actions_pi), dim=1  # type: ignore
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = ent_coef * log_prob - min_qf_pi
            # actor_loss = actor_loss * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            actor_loss = actor_loss.mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            # th.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor.optimizer.step()

            # Compute reward estimation loss

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            # Student update critic and actor, and update reward estimation
            student_replay_obs = replay_data.observations["observation"]  # type: ignore
            student_replay_next_obs = replay_data.next_observations["observation"]  # type: ignore
            # Action by the current actor for the sampled state
            student_actions_pi, student_log_prob = self.student_actor.action_log_prob(
                student_replay_obs
            )
            student_log_prob = student_log_prob.reshape(-1, 1)

            # For student agent's entropy coefficient
            # student_ent_coef_loss = None
            # if (
            #     self.student_ent_coef_optimizer is not None
            #     and self.log_student_ent_coef is not None
            # ):
            #     # Important: detach the variable from the graph
            #     # so we don't change it with other losses
            #     # see https://github.com/rail-berkeley/softlearning/issues/60
            #     student_ent_coef = th.exp(self.log_student_ent_coef.detach())
            #     # log_prob -> student_log_prob
            #     student_ent_coef_loss = -(
            #         self.log_student_ent_coef
            #         * (student_log_prob + self.target_entropy).detach()
            #     ).mean()
            #     student_ent_coef_losses.append(ent_coef_loss.item())  # type: ignore
            # else:
            #     student_ent_coef = self.student_ent_coef_tensor

            # student_ent_coefs.append(student_ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            # if (
            #     student_ent_coef_loss is not None
            #     and self.student_ent_coef_optimizer is not None
            # ):
            #     self.student_ent_coef_optimizer.zero_grad()
            #     student_ent_coef_loss.backward()
            #     self.student_ent_coef_optimizer.step()
            # self.student_ent_coef = self.ent_coef
            # Student update critic and actor, and update reward estimation

            # with th.no_grad():
            #     # Select action according to policy
            #     (
            #         student_next_actions,
            #         student_next_log_prob,
            #     ) = self.student_actor.action_log_prob(student_replay_next_obs)

            #     # Compute the next Q values: min over all critics targets
            #     student_next_q_values = th.cat(
            #         self.student_critic_target(
            #             student_replay_next_obs, student_next_actions
            #         ),
            #         dim=1,
            #     )
            #     student_next_q_values, _ = th.min(
            #         student_next_q_values, dim=1, keepdim=True
            #     )
            #     # add entropy term
            #     student_next_q_values = student_next_q_values - (
            #         self.student_gamma != 1
            #     ) * student_ent_coef * student_next_log_prob.reshape(-1, 1)
            #     # td error + entropy term
            #     student_target_q_values = (
            #         replay_data.rewards
            #         + (1 - replay_data.dones)
            #         * self.student_gamma
            #         * student_next_q_values
            #     )
            #     # handle average reward
            #     student_target_q_values -= (
            #         self.student_gamma == 1
            #     ) * replay_data.rewards.mean()

            # # Get current Q-values estimates for each critic network
            # # using action from the replay buffer
            # student_current_q_values = self.student_critic(
            #     student_replay_obs, replay_data.actions
            # )

            # # Compute critic loss
            # student_critic_loss = 0.5 * sum(
            #     F.mse_loss(student_current_q, student_target_q_values)
            #     for student_current_q in student_current_q_values
            # )

            # Conservative critic loss
            # random_actions, cql_loss = conservative_q_loss(
            #     self.student_critic,
            #     self.student_actor,
            #     importance_sampling_n=10,
            #     state=student_replay_obs,
            #     actions=student_actions_pi,
            #     next_state=student_replay_next_obs,
            #     action_space=self.action_space,
            #     cql_temp=1.0,
            #     q_pred=student_current_q_values,
            #     random_actions=self.random_actions,
            # )

            # self.random_actions = random_actions
            # student_critic_loss += cql_loss

            # assert isinstance(student_critic_loss, th.Tensor)  # for type checker
            # student_critic_losses.append(student_critic_loss.item())  # type: ignore[union-attr]

            # # Optimize the critic
            # self.student_critic.optimizer.zero_grad()
            # student_critic_loss.backward()
            # self.student_critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            # student_q_values_pi = th.cat(
            #     self.student_critic(student_replay_obs, student_actions_pi), dim=1
            # )
            # student_min_qf_pi, _ = th.min(student_q_values_pi, dim=1, keepdim=True)

            # Assymetric actor loss
            q_values_pi = th.cat(
                self.critic(replay_data.observations["state"], student_actions_pi),  # type: ignore
                dim=1,
            )
            teacher_min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            student_actor_loss = ent_coef * student_log_prob - teacher_min_qf_pi
            student_actor_loss = student_actor_loss.mean()
            student_actor_loss += F.mse_loss(student_actions_pi, actions_pi.detach())

            student_actor_losses.append(student_actor_loss.item())

            # Optimize the actor
            self.student_actor.optimizer.zero_grad()
            student_actor_loss.backward()
            self.student_actor.optimizer.step()
            student_average_reward_list.append(replay_data.rewards.mean())

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.student_critic.parameters(),
                    self.student_critic_target.parameters(),
                    self.tau,
                )
                # Copy running stats, see GH issue #996
                polyak_update(
                    self.student_batch_norm_stats,
                    self.student_batch_norm_stats_target,
                    1.0,
                )

        self._n_updates += gradient_steps

        # Adjust domain randomization scale
        if self.env.env_is_wrapped:  # type: ignore
            self.env.unwrapped.domain_randomization_scale = (  # type: ignore
                self.student_domain_randomization_scale
                * (1 - self._current_progress_remaining),
            )

        else:
            self.env.domain_randomization_scale = (  # type: ignore
                self.student_domain_randomization_scale
                * (1 - self._current_progress_remaining),
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/student_ent_coef", np.mean(student_ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/student_actor_loss", np.mean(student_actor_losses))
        self.logger.record("train/student_critic_loss", np.mean(student_critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def scale_tensor(self, tensor, new_min=-100, new_max=100):
        # Normalize the tensor to [0, 1]
        tensor = (tensor - th.min(tensor)) / (th.max(tensor) - th.min(tensor))
        # Scale to new range
        tensor = tensor * (new_max - new_min) + new_min
        return tensor

    def learn(
        self: SelfL2TRL,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "PMD",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfL2TRL:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "actor",
            "critic",
            "critic_target",
        ]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = [
            "policy",
            "actor.optimizer",
            "critic.optimizer",
            "student_policy",
            "student_actor.optimizer",
            "student_critic.optimizer",
        ]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def student_predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.student_policy.predict(  # type: ignore
            observation["observation"], state, episode_start, deterministic  # type: ignore
        )

    def teacher_predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(
            observation["state"], state, episode_start, deterministic
        )

    def load_replay_buffer_to(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ):
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.expert_replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(
            self.expert_replay_buffer, ReplayBuffer
        ), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(
            self.replay_buffer, "handle_timeout_termination"
        ):  # pragma: no cover
            self.expert_replay_buffer.handle_timeout_termination = False
            self.expert_replay_buffer.timeouts = np.zeros_like(
                self.expert_replay_buffer.dones
            )

        return self.expert_replay_buffer

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # self.explorer has epsilon chance to be student
        epsilon = 0.2
        if np.random.uniform() < epsilon and self.num_timesteps > 0:
            self.explorer = "student"
        else:
            self.explorer = "teacher"

        if self.explorer == "teacher":
            return self.policy.predict(
                observation["state"], state, episode_start, deterministic
            )
        else:
            return self.student_policy.predict(  # type: ignore
                observation["observation"], state, episode_start, deterministic  # type: ignore
            )
