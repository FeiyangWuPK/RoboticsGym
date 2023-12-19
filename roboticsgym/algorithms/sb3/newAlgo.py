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

from stable_baselines3.common.buffers import ReplayBuffer
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
from stable_baselines3.common.utils import update_learning_rate

from roboticsgym.algorithms.sb3.policies import (
    Actor,
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
    HIPPolicy,
)
from roboticsgym.algorithms.sb3.inversepolicies import (
    IPMDPolicy,
    CnnPolicy,
    MultiInputPolicy,
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

SelfHIP = TypeVar("SelfHIP", bound="HIP")


class HIP(OffPolicyAlgorithm):
    """
    Hallucinated Inverse Policy mirror descent algorithm (HIP)
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
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "HIPPolicy": HIPPolicy,
        "IPMDPolicy": IPMDPolicy,
    }
    policy: HIPPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[HIPPolicy], Type[IPMDPolicy]],
        env: Union[GymEnv, str],
        student_policy: Union[str, Type[IPMDPolicy]] = IPMDPolicy,
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
        expert_replaybuffer: str = "",
        expert_replaybuffersize: int = 6000,
        student_begin: int = int(0),
        reward_reg_param: float = 0.05,
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

        self.expert_replaybuffer = expert_replaybuffer
        self.expert_replaybuffersize = expert_replaybuffersize

        self.student_irl_begin_timesteps = student_begin

        self.reward_reg_param = reward_reg_param

        if _init_setup_model:
            self._setup_model()

    def _init_student_policy(
        self,
        student_policy: Union[str, Type[IPMDPolicy]] = IPMDPolicy,
        student_policy_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(student_policy, str):
            self.student_policy_class = self._get_policy_from_name(student_policy)
        else:
            self.student_policy_class = student_policy

        self.student_policy_kwargs = student_policy_kwargs

        # from off_policy_algorithm super()._setup_model()
        # partial_obversevation_space is from Environment's partial_observation_space
        # self.partial_observation_space = self.env.get_attr("partial_observation_space")[0]
        self.partial_observation_space = self.observation_space
        self.student_policy = (
            self.student_policy_class(  # pytype:disable=not-instantiable
                self.partial_observation_space,
                self.action_space,
                self.lr_schedule,
                **self.student_policy_kwargs,  # pytype:disable=not-instantiable
            )
        )
        self.student_policy = self.student_policy.to(self.device)

    def _setup_model(self) -> None:
        super()._setup_model()

        self.clip_range = get_schedule_fn(self.clip_range)
        self._init_student_policy(self.student_policy, self.student_policy_kwargs)
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

        self.expert_replay_buffer = ReplayBuffer(
            self.expert_replaybuffersize,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
        )
        self.load_replay_buffer_to(path=self.expert_replaybuffer)
        self.expert_replay_data = self.expert_replay_buffer._get_samples(
            np.arange(0, self.expert_replaybuffersize)
        )
        print("expert episodic reward", self.expert_replay_data.rewards.sum())

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
        if isinstance(self.policy, IPMDPolicy):
            print("Teacher using inverse RL")
            self.reward_est = self.policy.reward_est
        else:
            print("Teacher using Imitation Learning")

        self.student_actor = self.student_policy.actor
        self.student_critic = self.student_policy.critic
        self.student_critic_target = self.student_policy.critic_target
        self.student_reward_est = self.student_policy.reward_est

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.student_policy.set_training_mode(True)
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

        clip_range = self.clip_range(self._current_progress_remaining)

        # update_learning_rate(self.student_reward_est.optimizer, self.lr_schedule(self._current_progress_remaining) * 5)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        # For student agent
        student_actor_losses, student_critic_losses = [], []
        student_ent_coef_losses, student_ent_coefs = [], []
        average_reward_list = []
        reward_est_losses = []
        student_reward_est_losses = []
        student_reward_est_eval_losses = []
        student_average_reward_list = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            student_replay_obs = replay_data.observations
            # TODO: transform replay_data.obs to student's partial obs,
            # currently pretend they are the same agent

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
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
            if isinstance(self.policy, IPMDPolicy):
                estimated_rewards = th.cat(
                    self.reward_est(replay_data.observations, replay_data.actions),
                    dim=1,
                )
                estimated_rewards_copy = estimated_rewards.detach()
                self.estimated_average_reward = estimated_rewards_copy.mean()
                average_reward_list.append(self.estimated_average_reward.item())
            else:
                estimated_rewards_copy = replay_data.rewards

            # teacher update critic and actor
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - (
                    self.gamma != 1
                ) * ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = (
                    estimated_rewards_copy
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

                target_q_values -= (self.gamma == 1) * estimated_rewards_copy.mean()

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
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

            # Compute the ratio of old and new
            # _, log_prob_old = old_actor.action_log_prob(replay_data.observations)
            # ratio = th.exp(log_prob.detach() - log_prob_old.detach())

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
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

            if isinstance(self.policy, IPMDPolicy):
                # Get expert reward estimation
                expert_estimated_rewards = th.cat(
                    self.reward_est(
                        self.expert_replay_data.observations,
                        self.expert_replay_data.actions,
                    ),
                    dim=1,
                )
                estimated_rewards = th.cat(
                    self.reward_est(replay_data.observations, actions_pi.detach()),
                    dim=1,
                )
                alpha = 0.05
                reward_est_loss = (
                    estimated_rewards.mean()
                    - expert_estimated_rewards.mean()
                    + alpha
                    * (
                        th.linalg.norm(
                            th.cat([estimated_rewards, expert_estimated_rewards])
                        )
                    )
                )
                reward_est_losses.append(reward_est_loss.item())
                self.reward_est.optimizer.zero_grad()
                reward_est_loss.backward()
                self.reward_est.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            student_replay_obs = replay_data.observations
            # Action by the current actor for the sampled state
            student_actions_pi, student_log_prob = self.student_actor.action_log_prob(
                student_replay_obs
            )
            student_log_prob = student_log_prob.reshape(-1, 1)
            student_actions_pi_copy = student_actions_pi.detach()

            # For student agent
            student_ent_coef_loss = None
            if (
                self.student_ent_coef_optimizer is not None
                and self.log_student_ent_coef is not None
            ):
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                student_ent_coef = th.exp(self.log_student_ent_coef.detach())
                # log_prob -> student_log_prob
                student_ent_coef_loss = -(
                    self.log_student_ent_coef
                    * (student_log_prob + self.target_entropy).detach()
                ).mean()
                student_ent_coef_losses.append(ent_coef_loss.item())
            else:
                student_ent_coef = self.student_ent_coef_tensor

            student_ent_coefs.append(student_ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if (
                student_ent_coef_loss is not None
                and self.student_ent_coef_optimizer is not None
            ):
                self.student_ent_coef_optimizer.zero_grad()
                student_ent_coef_loss.backward()
                self.student_ent_coef_optimizer.step()
            self.student_ent_coef = self.ent_coef
            # Student update critic and actor, and update reward estimation
            student_replay_next_obs = replay_data.next_observations
            estimated_rewards = th.cat(
                self.student_reward_est(student_replay_obs, replay_data.actions), dim=1
            )
            if self.num_timesteps < int(self.student_irl_begin_timesteps):
                estimated_rewards_copy = replay_data.rewards
            else:
                estimated_rewards_copy = estimated_rewards.detach()
            self.estimated_average_reward = estimated_rewards_copy.mean()
            with th.no_grad():
                # Select action according to policy
                # student_next_actions, student_next_log_prob = self.student_actor.action_log_prob(student_replay_next_obs)
                student_next_actions, student_next_log_prob = (
                    next_actions,
                    next_log_prob,
                )
                # Compute the next Q values: min over all critics targets
                student_next_q_values = th.cat(
                    self.student_critic_target(
                        student_replay_next_obs, student_next_actions
                    ),
                    dim=1,
                )
                student_next_q_values, _ = th.min(
                    student_next_q_values, dim=1, keepdim=True
                )
                # add entropy term
                student_next_q_values = student_next_q_values - (
                    self.gamma != 1
                ) * student_ent_coef * student_next_log_prob.reshape(-1, 1)
                # td error + entropy term
                student_target_q_values = (
                    estimated_rewards_copy
                    + (1 - replay_data.dones)
                    * self.student_gamma
                    * student_next_q_values
                )
                # handle average reward
                student_target_q_values -= (
                    self.student_gamma == 1
                ) * estimated_rewards_copy.mean()

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            student_current_q_values = self.student_critic(
                student_replay_obs, replay_data.actions
            )

            # Compute critic loss
            student_critic_loss = 0.5 * sum(
                F.mse_loss(student_current_q, student_target_q_values)
                for student_current_q in student_current_q_values
            )
            assert isinstance(student_critic_loss, th.Tensor)  # for type checker
            student_critic_losses.append(student_critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.student_critic.optimizer.zero_grad()
            student_critic_loss.backward()
            self.student_critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            student_q_values_pi = th.cat(
                self.student_critic(student_replay_obs, student_actions_pi), dim=1
            )
            student_min_qf_pi, _ = th.min(student_q_values_pi, dim=1, keepdim=True)
            student_actor_loss = student_ent_coef * student_log_prob - student_min_qf_pi
            student_actor_loss += F.mse_loss(student_actions_pi, actions_pi.detach())
            student_actor_loss = student_actor_loss.mean()

            student_actor_losses.append(student_actor_loss.item())

            # Optimize the actor
            self.student_actor.optimizer.zero_grad()
            student_actor_loss.backward()
            # th.nn.utils.clip_grad_norm_(self.student_actor.parameters(), 0.5)
            self.student_actor.optimizer.step()

            # Get expert reward estimation
            # expert_replay_data = self.expert_replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
            expert_replay_data = self.expert_replay_data
            expert_estimated_rewards = th.cat(
                self.student_reward_est(
                    expert_replay_data.observations, expert_replay_data.actions
                ),
                dim=1,
            )

            student_average_reward_list.append(
                self.estimated_average_reward.cpu().numpy()
            )
            # if self.num_timesteps > self.student_irl_begin_timesteps:
            #     estimated_rewards = th.cat(self.student_reward_est(student_replay_obs, replay_data.actions), dim=1)
            #     student_reward_est_loss = estimated_rewards.mean() - expert_estimated_rewards.mean() + alpha*(th.linalg.norm(th.cat([estimated_rewards, expert_estimated_rewards])))
            # else:
            #     student_reward_est_loss = estimated_rewards.mean() - expert_estimated_rewards.mean() + alpha*(th.linalg.norm(th.cat([estimated_rewards, expert_estimated_rewards])))
            estimated_rewards = th.cat(
                self.student_reward_est(student_replay_obs, student_actions_pi_copy),
                dim=1,
            )
            # estimated_rewards = self.scale_tensor(estimated_rewards)
            # expert_estimated_rewards = self.scale_tensor(expert_estimated_rewards)
            student_reward_est_loss = (
                estimated_rewards.mean()
                - expert_estimated_rewards.mean()
                + self.reward_reg_param
                * (
                    th.linalg.norm(
                        th.cat([estimated_rewards, expert_estimated_rewards])
                    )
                )
            )
            student_reward_est_losses.append(student_reward_est_loss.item())

            estimated_rewards_teacher = th.cat(
                self.student_reward_est(student_replay_obs, replay_data.actions), dim=1
            )
            current_reward_est_loss = (
                estimated_rewards.mean()
                - estimated_rewards_teacher.mean()
                + self.reward_reg_param
                * (
                    th.linalg.norm(
                        th.cat([estimated_rewards, estimated_rewards_teacher])
                    )
                )
            )

            reward_est_loss = student_reward_est_loss
            # reward_est_loss = current_reward_est_loss
            # reward_est_loss = student_reward_est_loss + current_reward_est_loss

            if self.num_timesteps > self.student_irl_begin_timesteps:
                self.student_reward_est.optimizer.zero_grad()
                reward_est_loss.backward()
                self.student_reward_est.optimizer.step()

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

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/student_ent_coef", np.mean(student_ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/student_actor_loss", np.mean(student_actor_losses))
        self.logger.record("train/student_critic_loss", np.mean(student_critic_losses))
        self.logger.record("train/student_avg_est_loss", student_reward_est_losses[-1])
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def scale_tensor(self, tensor, new_min=-100, new_max=100):
        # Normalize the tensor to [0, 1]
        tensor = (tensor - th.min(tensor)) / (th.max(tensor) - th.min(tensor))
        # Scale to new range
        tensor = tensor * (new_max - new_min) + new_min
        return tensor

    def learn(
        self: SelfHIP,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHIP:
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
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
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
        return self.student_policy.predict(
            observation, state, episode_start, deterministic
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


class EvalStudentCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type"
                f"{self.training_env} != {self.eval_env}"
            )

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if (
            self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        ):  # and self.model.num_timesteps > self.model.student_irl_begin_timesteps:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_student_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Student Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/student_mean_reward", float(mean_reward))
            self.logger.record("eval/student_mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


def evaluate_student_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.student_predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
