import mujoco
from typing import Callable
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
)

from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
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
)
from roboticsgym.algorithms.sb3.utilities import (
    evaluate_student_policy,
    evaluate_teacher_policy,
    EvalStudentCallback,
    EvalTeacherCallback,
    linear_schedule,
    VideoEvalCallback,
)

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="train_digit")
def train_digit_sac(cfg: DictConfig):
    """
    Train Digit with SAC.
    """
    run = wandb.init(
        project=cfg.wandb.project,
        config=dict(cfg),  # Passes all the configurations to WandB
        name=cfg.wandb.run_name,
        monitor_gym=cfg.env.name,
        save_code=True,
        group=cfg.wandb.group,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        # entity=cfg.wandb.entity,
    )

    # Create the environment
    env = make_vec_env(
        cfg.env.name,
        n_envs=cfg.env.n_envs,
        seed=cfg.env.seed,
    )
    eval_env = make_vec_env(
        cfg.env.name,
        n_envs=1,
        seed=cfg.env.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/{run.id}/",  # type: ignore
        log_path=f"logs/{run.project}/{run.name}/{run.id}/",  # type: ignore
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=False,
        render=True,
        verbose=1,
    )
    video_callback = VideoEvalCallback(eval_every=10000, eval_env=eval_env)
    wandb_callback = WandbCallback()

    # Create the model
    # model = SAC(
    #     "MlpPolicy",
    #     env,
    #     verbose=cfg.training.verbose,
    #     # learning_rate=cfg.training.learning_rate,
    #     learning_rate=linear_schedule(3e-3),
    #     buffer_size=cfg.training.buffer_size,
    #     batch_size=cfg.training.batch_size,
    #     learning_starts=cfg.training.learning_starts,
    #     train_freq=cfg.training.train_freq,
    #     gradient_steps=cfg.training.gradient_steps,
    #     ent_coef=cfg.training.ent_coef,
    #     tensorboard_log=f"logs/{run.project}/{run.name}/{run.id}/",  # Log to WandB directory # type: ignore
    # )
    # model.set_parameters(
    #     "logs/CoRL2024 L2T Digit/With old PD gain/o00dyabi/best_model.zip"
    # )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=cfg.training.verbose,
        learning_rate=linear_schedule(3e-3),
        tensorboard_log=f"logs/{run.project}/{run.name}/{run.id}/",  # Log to WandB directory # type: ignore
    )
    # Train the model
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        progress_bar=True,
        log_interval=100,
        callback=CallbackList([eval_callback, video_callback, wandb_callback]),
    )

    run.finish()  # type: ignore


@hydra.main(config_path="configs", config_name="train_digit")
def train_digit_ppo(cfg: DictConfig):
    """
    Train Digit with PPO.
    """
    run = wandb.init(
        project=cfg.wandb.project,
        config=dict(cfg),  # Passes all the configurations to WandB
        name=cfg.wandb.run_name,
        monitor_gym=cfg.env.name,
        save_code=True,
        group=cfg.wandb.group,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        # entity=cfg.wandb.entity,
    )

    # Create the environment
    env = make_vec_env(
        cfg.env.name,
        n_envs=cfg.env.n_envs,
        seed=cfg.env.seed,
        # env_kwargs={"render_mode": "human"},
    )
    eval_env = make_vec_env(
        cfg.env.name,
        n_envs=1,
        seed=cfg.env.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/{run.id}/",  # type: ignore
        log_path=f"logs/{run.project}/{run.name}/{run.id}/",  # type: ignore
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=False,
        render=True,
        verbose=1,
    )
    video_callback = VideoEvalCallback(eval_every=500000, eval_env=eval_env)
    wandb_callback = WandbCallback()

    # Create the model
    # model = SAC(
    #     "MlpPolicy",
    #     env,
    #     verbose=cfg.training.verbose,
    #     # learning_rate=cfg.training.learning_rate,
    #     learning_rate=linear_schedule(3e-3),
    #     buffer_size=cfg.training.buffer_size,
    #     batch_size=cfg.training.batch_size,
    #     learning_starts=cfg.training.learning_starts,
    #     train_freq=cfg.training.train_freq,
    #     gradient_steps=cfg.training.gradient_steps,
    #     ent_coef=cfg.training.ent_coef,
    #     tensorboard_log=f"logs/{run.project}/{run.name}/{run.id}/",  # Log to WandB directory # type: ignore
    # )
    # model.set_parameters(
    #     "logs/CoRL2024 L2T Digit/With old PD gain/o00dyabi/best_model.zip"
    # )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=cfg.training.verbose,
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        tensorboard_log=f"logs/{run.project}/{run.name}/{run.id}/",  # Log to WandB directory # type: ignore
    )
    model.set_parameters(
        "logs/CoRL2024 L2T Digit/PPO With new PD gain, with gear ratio, reduce control cost/vhppt5mf/best_model.zip"
    )
    # Train the model
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        progress_bar=True,
        log_interval=100,
        callback=CallbackList([eval_callback, video_callback, wandb_callback]),
    )

    run.finish()  # type: ignore


@hydra.main(config_path="configs", config_name="train_digit")
def visualize_expert_trajectory(cfg: DictConfig):
    """
    Visualize Digit Expert data.
    """
    run = wandb.init(
        project=cfg.wandb.project,
        config=dict(cfg),  # Passes all the configurations to WandB
        name="Visualize expert trajectory",
        monitor_gym=cfg.env.name,
        save_code=True,
        group=cfg.wandb.group,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        # entity=cfg.wandb.entity,
        mode="offline",
    )

    # Create the environment
    env = make_vec_env(
        "DigitViz-v1",
        n_envs=1,
        seed=cfg.env.seed,
        # env_kwargs={"render_mode": "human"},
    )
    eval_env = make_vec_env(
        "DigitViz-v1",
        n_envs=1,
        seed=cfg.env.seed,
        env_kwargs={"render_mode": "human"},
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/{run.id}/",  # type: ignore
        log_path=f"logs/{run.project}/{run.name}/{run.id}/",  # type: ignore
        eval_freq=1,
        n_eval_episodes=5,
        deterministic=True,
        render=True,
        verbose=1,
    )
    video_callback = VideoEvalCallback(eval_every=1, eval_env=eval_env)
    wandb_callback = WandbCallback()

    # Create the model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=cfg.training.verbose,
        learning_rate=cfg.training.learning_rate,
        buffer_size=cfg.training.buffer_size,
        batch_size=cfg.training.batch_size,
        learning_starts=cfg.training.learning_starts,
        train_freq=cfg.training.train_freq,
        gradient_steps=cfg.training.gradient_steps,
        ent_coef=cfg.training.ent_coef,
        tensorboard_log=f"logs/{run.project}/{run.name}/{run.id}/",  # Log to WandB directory # type: ignore
    )

    # Train the model
    model.learn(
        total_timesteps=1,
        progress_bar=True,
        log_interval=10,
        callback=CallbackList([eval_callback, video_callback, wandb_callback]),
    )

    run.finish()  # type: ignore
