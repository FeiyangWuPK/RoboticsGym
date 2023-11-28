# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional, Union, Tuple
from numbers import Number
import torch
from torch import Tensor
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
    CompositeSpec,
    TensorSpec,
)
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.objectives.sac import DiscreteSACLoss
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator
from torchrl.objectives.utils import (
    ValueEstimators,
    default_value_kwargs,
)
from tensordict.tensordict import TensorDictBase

SHAPE_ERR = (
    "All input tensors (value, reward and done states) must share a unique shape."
)
# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu"):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
            )
    elif lib == "dm_control":
        env = DMControlEnv(cfg.env.name, cfg.env.task)
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def apply_env_transforms(env, max_episode_steps=1000):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg):
    """Make environments for training and evaluation."""
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(lambda cfg=cfg: env_maker(cfg)),
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(lambda cfg=cfg: env_maker(cfg)),
        ),
        train_env.transform.clone(),
    )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    buffer_scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


# ====================================================================
# Model
# -----


# Rewrite td0_return_estimate for considering average-reward MDPs
def PMDtd0_return_estimate(
    gamma: float,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor | None = None,
    *,
    done: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # noqa: D417
    """TD(0) discounted return estimate of a trajectory.

    Also known as bootstrapped Temporal Difference or one-step return.

    Args:
        gamma (scalar): exponential mean discount.
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.

    Keyword Args:
        done (Tensor): Deprecated. Use ``terminated`` instead.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if done is not None and terminated is None:
        terminated = done
        warnings.warn(
            "done for td0_return_estimate is deprecated. Pass ``terminated`` instead."
        )
    if not (next_state_value.shape == reward.shape == terminated.shape):
        raise RuntimeError(SHAPE_ERR)
    not_terminated = (~terminated).int()
    if_average_reward = gamma == 1
    advantage = (
        reward
        - reward.mean() * if_average_reward
        + gamma * not_terminated * next_state_value
    )
    return advantage


# Rewrite TD0Estimator
class PMDTD0Estimator(TD0Estimator):
    """
    The only difference is that we need a new td0_return_estimate to
    incoporate average-reward MDPs
    """

    def value_estimate(
        self,
        tensordict,
        target_params: Optional[TensorDictBase] = None,
        next_value: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        gamma = self.gamma.to(device)
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.average_rewards:
            reward = reward - reward.mean()
            reward = reward / reward.std().clamp_min(1e-4)
            tensordict.set(
                ("next", self.tensor_keys.reward), reward
            )  # we must update the rewards if they are used later in the code
        if next_value is None:
            next_value = self._next_value(tensordict, target_params, kwargs=kwargs)

        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        value_target = PMDtd0_return_estimate(
            gamma=gamma,
            next_state_value=next_value,
            reward=reward,
            done=done,
            terminated=terminated,
        )
        return value_target


# TODO: make a new pmd loss module to handle the average-reward case
# and the inverse RL reward loss
class PMDLoss(SACLoss):
    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        value_network: Optional[TensorDictModule] = None,
        *,
        num_qvalue_nets: int = 2,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = None,
        max_alpha: float = None,
        action_spec=None,
        fixed_alpha: bool = False,
        target_entropy: Union[str, float] = "auto",
        delay_actor: bool = False,
        delay_qvalue: bool = True,
        delay_value: bool = True,
        gamma: float = None,
        priority_key: str = None,
        separate_losses: bool = False,
    ) -> None:
        SACLoss.__init__(
            self,
            actor_network,
            qvalue_network,
            value_network=value_network,
            num_qvalue_nets=num_qvalue_nets,
            loss_function=loss_function,
            alpha_init=alpha_init,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            action_spec=action_spec,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
            delay_value=delay_value,
            gamma=gamma,
            priority_key=priority_key,
            separate_losses=separate_losses,
        )

    def _compute_target_v2(self, tensordict) -> Tensor:
        r"""Value network for SAC v2.

        SAC v2 is based on a value estimate of the form:

        .. math::

          V = Q(s,a) - \alpha * \log p(a | s)

        This class computes this value given the actor and qvalue network

        """
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            with set_exploration_type(ExplorationType.RANDOM):
                next_tensordict = tensordict.get("next").clone(False)
                next_dist = self.actor_network.get_dist(
                    next_tensordict, params=self.actor_network_params
                )
                next_action = next_dist.rsample()
                next_tensordict.set(self.tensor_keys.action, next_action)
                next_sample_log_prob = next_dist.log_prob(next_action)

            # get q-values
            next_tensordict_expand = self._vmap_qnetworkN0(
                next_tensordict, self.target_qvalue_network_params
            )
            state_action_value = next_tensordict_expand.get(
                self.tensor_keys.state_action_value
            )
            if (
                state_action_value.shape[-len(next_sample_log_prob.shape) :]
                != next_sample_log_prob.shape
            ):
                next_sample_log_prob = next_sample_log_prob.unsqueeze(-1)
            if self.gamma == 1:
                # in average-reward policy evaluation, no entropy term is needed
                next_state_value = state_action_value
            else:
                next_state_value = (
                    state_action_value - self._alpha * next_sample_log_prob
                )
            next_state_value = next_state_value.min(0)[0]
            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
            return target_value

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        if self._version == 1:
            value_net = self.actor_critic
        elif self._version == 2:
            # we will take care of computing the next value inside this module
            value_net = None
        else:
            # unreachable
            raise NotImplementedError

        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=value_net,
            )
        elif value_type is ValueEstimators.TD0:
            # change to PMD TD0 estimator to consider average-reward MDPs
            self._value_estimator = PMDTD0Estimator(
                **hp,
                value_network=value_net,
            )
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=value_net,
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value_target": "value_target",
            "value": self.tensor_keys.value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)


class DiscretePMDLoss(DiscreteSACLoss):
    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        *,
        action_space: Union[str, TensorSpec] = None,
        num_actions: Optional[int] = None,
        num_qvalue_nets: int = 2,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = None,
        max_alpha: float = None,
        fixed_alpha: bool = False,
        target_entropy_weight: float = 0.98,
        target_entropy: Union[str, Number] = "auto",
        delay_qvalue: bool = True,
        priority_key: str = None,
        separate_losses: bool = False,
    ) -> None:
        DiscreteSACLoss.__init__(
            self,
            actor_network,
            qvalue_network,
            action_space=action_space,
            num_actions=num_actions,
            num_qvalue_nets=num_qvalue_nets,
            loss_function=loss_function,
            alpha_init=alpha_init,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            fixed_alpha=fixed_alpha,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            delay_qvalue=delay_qvalue,
            priority_key=priority_key,
            separate_losses=separate_losses,
        )

    def _compute_target(self, tensordict) -> Tensor:
        r"""Value network for SAC v2.

        SAC v2 is based on a value estimate of the form:

        .. math::

          V = Q(s,a) - \alpha * \log p(a | s)

        This class computes this value given the actor and qvalue network

        """
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            next_tensordict = tensordict.get("next").clone(False)

            # get probs and log probs for actions computed from "next"
            next_dist = self.actor_network.get_dist(
                next_tensordict, params=self.actor_network_params
            )
            next_prob = next_dist.probs
            next_log_prob = torch.log(torch.where(next_prob == 0, 1e-8, next_prob))

            # get q-values for all actions
            next_tensordict_expand = self._vmap_qnetworkN0(
                next_tensordict, self.target_qvalue_network_params
            )
            next_action_value = next_tensordict_expand.get(
                self.tensor_keys.action_value
            )

            # like in continuous SAC, we take the minimum of the value ensemble and subtract the entropy term
            if self.gamma == 1:
                next_state_value = next_action_value.min(0)[0]
            else:
                next_state_value = (
                    next_action_value.min(0)[0] - self._alpha * next_log_prob
                )
            # unlike in continuous SAC, we can compute the exact expectation over all discrete actions
            next_state_value = (next_prob * next_state_value).sum(-1).unsqueeze(-1)

            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
            return target_value

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=None,
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = PMDTD0Estimator(
                **hp,
                value_network=None,
            )
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=None,
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value": self.tensor_keys.value,
            "value_target": "value_target",
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)


def make_sac_agent(cfg, train_env, eval_env, device):
    """Make SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    actor_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.low,
        "max": action_spec.space.high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
        scale_lb=cfg.network.scale_lb,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 1,
        "activation_class": get_activation(cfg),
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model, model[0]


# TODO: make reward estimate
# def make_reward_estimate(cfg, train_env, eval_env, device)

# ====================================================================
# SAC Loss
# ---------


def make_loss_module(cfg, model, discrete_action: bool = False):
    """Make loss module and target network updater."""
    # Create SAC loss
    if discrete_action is True:
        loss_module = DiscretePMDLoss(
            actor_network=model[0],
            qvalue_network=model[1],
            num_qvalue_nets=2,
            loss_function=cfg.optim.loss_function,
            delay_actor=False,
            delay_qvalue=True,
            alpha_init=cfg.optim.alpha_init,
        )
    else:
        loss_module = PMDLoss(
            actor_network=model[0],
            qvalue_network=model[1],
            num_qvalue_nets=2,
            loss_function=cfg.optim.loss_function,
            delay_actor=False,
            delay_qvalue=True,
            alpha_init=cfg.optim.alpha_init,
        )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def split_critic_params(critic_params):
    critic1_params = []
    critic2_params = []

    for param in critic_params:
        data1, data2 = param.data.chunk(2, dim=0)
        critic1_params.append(nn.Parameter(data1))
        critic2_params.append(nn.Parameter(data2))
    return critic1_params, critic2_params


def make_sac_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError
