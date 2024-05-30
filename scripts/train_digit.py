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


from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecMonitor
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
from roboticsgym.algorithms.sb3.l2t_rl_onpolicy import L2TRL_O

from roboticsgym.algorithms.sb3.utilities import (
    evaluate_student_policy,
    evaluate_teacher_policy,
    EvalStudentCallback,
    EvalTeacherCallback,
    linear_schedule,
    VideoEvalCallback,
    StudentVideoEvalCallback,
)

import hydra
from omegaconf import DictConfig, OmegaConf
import datetime


@hydra.main(config_path="configs", config_name="train_digit_v1", version_base=None)
def train_digit_sac(cfg: DictConfig):
    """
    Train Digit with SAC.
    """
    import os

    os.environ["WANDB_SILENT"] = "true"

    run = wandb.init(
        project=cfg.wandb.project,
        config=dict(cfg),  # Passes all the configurations to WandB
        name=cfg.wandb.run_name,
        monitor_gym=cfg.env.name,
        # save_code=True,
        group=cfg.wandb.group,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        # entity=cfg.wandb.entity,
        # mode="offline",
        # notes="new mujoco, new ref traj",
    )
    # Convert unix time to human readable format
    start_time = datetime.datetime.fromtimestamp(run.start_time).strftime(
        "%Y-%m-%d-%H-%M-%S"
    )

    # Create the environment
    env = make_vec_env(
        cfg.env.name,
        n_envs=cfg.env.n_envs,
        seed=cfg.env.seed,
        vec_env_cls=SubprocVecEnv,
        # env_kwargs={"render_mode": "human"},
    )

    eval_env = make_vec_env(
        cfg.env.name,
        n_envs=1,
        seed=cfg.env.seed,
        vec_env_cls=SubprocVecEnv,
        # env_kwargs={"render_mode": "human"},
    )

    video_callback = VideoEvalCallback(eval_every=1, eval_env=eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/",  # type: ignore
        log_path=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/",  # type: ignore
        eval_freq=20000,
        n_eval_episodes=3,
        callback_on_new_best=video_callback,
        deterministic=True,
        render=True,
        verbose=1,
    )

    wandb_callback = WandbCallback()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=cfg.training.verbose,
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        train_freq=cfg.training.train_freq,
        gradient_steps=cfg.training.gradient_steps,
        tensorboard_log=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/",  # Log to WandB directory # type: ignore
    )

    # Train the model
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        progress_bar=True,
        log_interval=10,
        callback=CallbackList([eval_callback, wandb_callback]),
    )

    run.finish()  # type: ignore


@hydra.main(config_path="configs", config_name="train_digit_v1", version_base=None)
def train_digit_ppo(cfg: DictConfig):
    """
    Train Digit with PPO.
    """
    import os

    os.environ["WANDB_SILENT"] = "true"

    run = wandb.init(
        project=cfg.wandb.project,
        config=dict(cfg),  # Passes all the configurations to WandB
        name=cfg.wandb.run_name,
        monitor_gym=cfg.env.name,
        # save_code=True,
        group=cfg.wandb.group,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        # entity=cfg.wandb.entity,
        # mode="offline",
        # notes="new mujoco, new ref traj",
    )
    # Convert unix time to human readable format
    start_time = datetime.datetime.fromtimestamp(run.start_time).strftime(
        "%Y-%m-%d-%H-%M-%S"
    )

    # Create the environment
    env = make_vec_env(
        cfg.env.name,
        n_envs=cfg.env.n_envs,
        seed=cfg.env.seed,
        vec_env_cls=SubprocVecEnv,
        # env_kwargs={"render_mode": "human"},
    )

    eval_env = make_vec_env(
        cfg.env.name,
        n_envs=1,
        seed=cfg.env.seed,
        vec_env_cls=SubprocVecEnv,
        # env_kwargs={"render_mode": "human"},
    )

    video_callback = VideoEvalCallback(eval_every=1, eval_env=eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/",  # type: ignore
        log_path=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/",  # type: ignore
        eval_freq=20000,
        n_eval_episodes=3,
        callback_on_new_best=video_callback,
        deterministic=True,
        render=True,
        verbose=1,
    )

    wandb_callback = WandbCallback()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=cfg.training.verbose,
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        tensorboard_log=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/",  # Log to WandB directory # type: ignore
    )

    # model.set_parameters(
    #     "logs/CoRL2024 L2T Digit/FKHY-v1/2024-05-25-15-27-20-su08c4uk/best_model.zip"
    # )

    # Train the model
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        progress_bar=True,
        log_interval=10,
        callback=CallbackList([eval_callback, wandb_callback]),
    )

    run.finish()  # type: ignore


@hydra.main(config_path="configs", config_name="train_digit", version_base=None)
def visualize_expert_trajectory(cfg: DictConfig):
    """
    Visualize Digit Expert data.
    """
    run = wandb.init(
        project=cfg.wandb.project,
        config=dict(cfg),  # Passes all the configurations to WandB
        name="Visualize expert trajectory",
        monitor_gym=True,
        save_code=True,
        group=cfg.wandb.group,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        # entity=cfg.wandb.entity,
        # mode="offline",
    )

    # Create the environment
    env = make_vec_env(
        "DigitFKHY-v1",
        n_envs=1,
        seed=cfg.env.seed,
        # env_kwargs={"render_mode": "human"},
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = make_vec_env(
        "DigitFKHY-v1",
        n_envs=1,
        seed=cfg.env.seed,
        env_kwargs={"render_mode": "human"},
        vec_env_cls=SubprocVecEnv,
    )

    # video_env = VecVideoRecorder(
    #     eval_env,
    #     "./videos/",
    #     record_video_trigger=lambda x: x % 1 == 0,
    #     video_length=2000,
    # )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/{run.id}/",  # type: ignore
        log_path=f"logs/{run.project}/{run.name}/{run.id}/",  # type: ignore
        eval_freq=1,
        n_eval_episodes=1,
        deterministic=True,
        render=True,
        verbose=1,
    )
    # video_callback = VideoEvalCallback(eval_every=1, eval_env=eval_env)
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
        callback=CallbackList([eval_callback, wandb_callback]),
    )

    run.finish()  # type: ignore


@hydra.main(config_path="configs", config_name="train_digit_v2", version_base=None)
def train_digit_L2TRL(cfg: DictConfig):
    """
    Train Digit with L2T.
    """

    import os

    os.environ["WANDB_SILENT"] = "true"
    env_kwargs = {
        "domain_randomization_scale": cfg.env.domain_randomization_scale,
        "render_mode": "rgb_array",
    }
    run = wandb.init(
        project=cfg.wandb.project,
        config=dict(cfg),  # Passes all the configurations to WandB
        name=cfg.wandb.run_name,
        monitor_gym=cfg.env.name,
        # save_code=True,
        group=cfg.wandb.group,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        # entity=cfg.wandb.entity,
        # mode="offline",
        # notes="new mujoco, new ref traj",
    )
    # Convert unix time to human readable format
    start_time = datetime.datetime.fromtimestamp(run.start_time).strftime(
        "%Y-%m-%d-%H-%M-%S"
    )

    # Create the environment
    env = make_vec_env(
        cfg.env.name,
        n_envs=cfg.env.n_envs,
        seed=cfg.env.seed,
        vec_env_cls=SubprocVecEnv,
        # env_kwargs={"render_mode": "human"},
        env_kwargs=env_kwargs,
    )

    teacher_eval_env = make_vec_env(
        cfg.env.name,
        n_envs=1,
        seed=cfg.env.seed,
        # vec_env_cls=SubprocVecEnv,
        # env_kwargs={"render_mode": "human"},
    )

    teacher_video_callback = VideoEvalCallback(
        eval_every=1, eval_env=teacher_eval_env, sub_prefix="teacher"
    )

    teacher_eval_callback = EvalCallback(
        teacher_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/teacher/",  # type: ignore
        log_path=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/teacher/",  # type: ignore
        eval_freq=10000,
        n_eval_episodes=1,
        callback_on_new_best=teacher_video_callback,
        deterministic=True,
        render=False,
        verbose=1,
    )

    student_eval_env = make_vec_env(
        cfg.env.name,
        n_envs=1,
        seed=cfg.env.seed,
        # vec_env_cls=SubprocVecEnv,
        # env_kwargs={"render_mode": "human"},
        env_kwargs=env_kwargs,
    )

    student_video_callback = StudentVideoEvalCallback(
        eval_every=1, eval_env=student_eval_env, sub_prefix="student"
    )

    student_eval_callback = EvalStudentCallback(
        student_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/student/",  # type: ignore
        log_path=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/student/",  # type: ignore
        eval_freq=10000,
        n_eval_episodes=1,
        callback_on_new_best=student_video_callback,
        deterministic=True,
        render=False,
        verbose=1,
    )

    wandb_callback = WandbCallback()

    model = L2TRL_O(
        "L2TPolicy",
        env,
        student_policy="L2TPolicy",
        verbose=cfg.training.verbose,
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        tensorboard_log=f"logs/{run.project}/{run.name}/{start_time}-{run.id}/",  # Log to WandB directory # type: ignore
        mixture_coeff=0.2,
    )
    model.set_parameters(
        "logs/CoRL2024 L2T Digit/L2T 200Mil/2024-05-27-19-25-59-1usag0g1/teacher/best_model.zip"
    )
    # validating the new version
    # model = PPO.load(
    #     env=env,
    #     path="logs/CoRL2024 L2T Digit/FKHY-v1/2024-05-25-15-27-20-su08c4uk/best_model.zip",
    # )

    # Train the model
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        progress_bar=True,
        log_interval=1,
        callback=CallbackList(
            [teacher_eval_callback, student_eval_callback, wandb_callback]
        ),
    )

    run.finish()  # type: ignore
