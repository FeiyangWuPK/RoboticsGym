import mujoco
from typing import Callable
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
)
from stable_baselines3.sac import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import wandb

from wandb.integration.sb3 import WandbCallback
import warnings

import roboticsgym.envs

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

from gymnasium.envs.registration import register
from roboticsgym.algorithms.sb3.l2t_rl import (
    L2TRL,
    evaluate_student_policy,
    evaluate_teacher_policy,
    EvalStudentCallback,
    EvalTeacherCallback,
)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train_digit_sac():
    """
    Train Digit with SAC.
    """

    # Create the environment
    env = make_vec_env("Digit-v1", n_envs=8)

    # Create the model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(3e-3),
        tensorboard_log="./logs/digit_sac/",
    )

    # Train the model
    model.learn(total_timesteps=int(1e7), log_interval=1000, progress_bar=True)
