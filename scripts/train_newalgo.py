import mujoco
from typing import Callable
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
)
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

from roboticsgym.algorithms.sb3.newAlgo import (
    HIP,
    EvalStudentCallback,
    evaluate_student_policy,
    EvalTeacherCallback,
    evaluate_teacher_policy,
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


def train_cassie_v4():
    config = {
        "policy_type": "IPMDPolicy",
        "total_timesteps": 5e6,
        "env_id": "CassieMirror-v4",
        "buffer_size": 1000000,
        "train_freq": 3,
        "gradient_steps": 3,
        "progress_bar": True,
        "verbose": 0,
        "ent_coef": "auto",
        "student_ent_coef": "auto",
        "learning_rate": linear_schedule(3e-3),
        "n_envs": 24,
        "batch_size": 300,
        "seed": 42,
        "expert_replaybuffersize": 600,
        "expert_replaybuffer": "expert_trajectories/cassie_v4/10traj_morestable",
        "student_begin": int(0),
        "teacher_gamma": 1.00,
        "student_gamma": 1.00,
        "reward_reg_param": 0.05,
        "student_domain_randomization_scale": 0.05,
        "explorer": "student",
    }
    run = wandb.init(
        project="ICML2024 Guided Learning",
        config=config,
        name=config["env_id"] + f'-{time.strftime("%Y-%m-%d-%H-%M-%S")}',
        tags=[config["env_id"]],
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        reinit=True,
        notes="student as actor, teacher learn from offline data",
        # mode="offline",
    )
    wandb.run.log_code(".")

    wandbcallback = WandbCallback(
        # gradient_save_freq=5000,
    )
    # Create log dir
    train_env = make_vec_env(
        config["env_id"], n_envs=config["n_envs"], vec_env_cls=SubprocVecEnv
    )
    # Separate evaluation env
    eval_env = make_vec_env(config["env_id"], n_envs=1, vec_env_cls=SubprocVecEnv)
    # Use deterministic actions for evaluation
    eval_callback = EvalTeacherCallback(
        eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/teacher/",
        log_path=f"logs/{run.project}/{run.name}/teacher/",
        eval_freq=10000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=1,
    )
    student_eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        # env_kwargs={
        #     "domain_randomization_scale": config["student_domain_randomization_scale"]
        # },
    )
    eval_student_callback = EvalStudentCallback(
        student_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/student/",
        log_path=f"logs/{run.project}/{run.name}/student/",
        eval_freq=10000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callback_list = CallbackList([eval_callback, wandbcallback, eval_student_callback])
    # Init model
    irl_model = HIP(
        policy=config["policy_type"],
        student_policy=config["policy_type"],
        env=train_env,
        gamma=config["teacher_gamma"],
        verbose=config["verbose"],
        student_gamma=config["student_gamma"],
        buffer_size=config["buffer_size"],
        ent_coef=config["ent_coef"],
        student_ent_coef=config["student_ent_coef"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        gradient_steps=config["gradient_steps"],
        expert_replaybuffer=config["expert_replaybuffer"],
        expert_replaybuffersize=config["expert_replaybuffersize"],
        tensorboard_log=f"logs/tensorboard/{run.name}/",
        seed=config["seed"],
        learning_starts=100,
        student_begin=config["student_begin"],
        reward_reg_param=config["reward_reg_param"],
        student_domain_randomization_scale=config["student_domain_randomization_scale"],
        explorer=config["explorer"],
    )

    # Model learning
    irl_model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback_list,
        progress_bar=config["progress_bar"],
        log_interval=50,
    )
    # Evaluation
    _, _ = evaluate_policy(irl_model, eval_env, n_eval_episodes=10)
    _, _ = evaluate_student_policy(irl_model, eval_env, n_eval_episodes=10)

    # Finish wandb run
    run.finish()


def train_cassie_v5():
    config = {
        "policy_type": "IPMDPolicy",
        "total_timesteps": 5e6,
        "env_id": "CassieMirror-v5",
        "buffer_size": 200000,
        "train_freq": 3,
        "gradient_steps": 3,
        "progress_bar": True,
        "verbose": 0,
        "ent_coef": "auto",
        "student_ent_coef": "auto",
        "learning_rate": linear_schedule(3e-3),
        "n_envs": 24,
        "batch_size": 300,
        "seed": 42,
        "expert_replaybuffersize": 600,
        "expert_replaybuffer": "expert_trajectories/cassie_v4/10traj_morestable.pkl",
        "student_begin": int(0),
        "teacher_gamma": 1.00,
        "student_gamma": 1.00,
        "reward_reg_param": 0.05,
        "student_domain_randomization_scale": 0.1,
        "explorer": "teacher",
        "state_only": False,
    }
    run = wandb.init(
        project="ICML2024 Guided Learning",
        config=config,
        # name=config["env_id"] + f'-{time.strftime("%Y-%m-%d-%H-%M-%S")}',
        name="Student using teacher reward",
        tags=[config["env_id"]],
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        reinit=True,
        notes="Imitating POMDP 0.1 w CL",
        # mode="offline",
    )
    wandb.run.log_code(".")

    wandbcallback = WandbCallback(
        # gradient_save_freq=5000,
    )
    # Create log dir
    train_env = make_vec_env(
        config["env_id"],
        n_envs=config["n_envs"],
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )
    # Separate evaluation env
    teacher_eval_env = make_vec_env(
        config["env_id"], n_envs=1, vec_env_cls=SubprocVecEnv
    )
    # Use deterministic actions for evaluation
    teacher_eval_callback = EvalTeacherCallback(
        teacher_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/teacher/",
        log_path=f"logs/{run.project}/{run.name}/teacher/",
        eval_freq=10000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=1,
    )
    student_eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )
    eval_student_callback = EvalStudentCallback(
        student_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/student/",
        log_path=f"logs/{run.project}/{run.name}/student/",
        eval_freq=10000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callback_list = CallbackList(
        [teacher_eval_callback, wandbcallback, eval_student_callback]
    )
    # Init model
    irl_model = HIP(
        policy=config["policy_type"],
        student_policy=config["policy_type"],
        env=train_env,
        gamma=config["teacher_gamma"],
        verbose=config["verbose"],
        student_gamma=config["student_gamma"],
        buffer_size=config["buffer_size"],
        ent_coef=config["ent_coef"],
        student_ent_coef=config["student_ent_coef"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        gradient_steps=config["gradient_steps"],
        expert_replaybuffer=config["expert_replaybuffer"],
        expert_replaybuffersize=config["expert_replaybuffersize"],
        tensorboard_log=f"logs/tensorboard/{run.name}/",
        seed=config["seed"],
        learning_starts=100,
        student_begin=config["student_begin"],
        reward_reg_param=config["reward_reg_param"],
        student_domain_randomization_scale=config["student_domain_randomization_scale"],
        explorer=config["explorer"],
        teacher_state_only_reward=config["state_only"],
    )

    # Model learning
    irl_model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback_list,
        progress_bar=config["progress_bar"],
        log_interval=50,
    )

    # Finish wandb run
    run.finish()


def test_student_policy():
    config = {
        "policy_type": "IPMDPolicy",
        "total_timesteps": 5e6,
        "env_id": "CassieMirror-v5",
        "buffer_size": 200000,
        "train_freq": 3,
        "gradient_steps": 3,
        "progress_bar": True,
        "verbose": 0,
        "ent_coef": "auto",
        "student_ent_coef": "auto",
        "learning_rate": linear_schedule(3e-3),
        "n_envs": 24,
        "batch_size": 300,
        "seed": 42,
        "expert_replaybuffersize": 600,
        "expert_replaybuffer": "expert_trajectories/cassie_v4/10traj_morestable.pkl",
        "student_begin": int(0),
        "teacher_gamma": 1.00,
        "student_gamma": 1.00,
        "reward_reg_param": 0.05,
        "student_domain_randomization_scale": 0.1,
        "explorer": "student",
        "state_only": False,
    }

    # Separate evaluation env
    eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )

    # Init model
    irl_model = HIP(
        policy=config["policy_type"],
        student_policy=config["policy_type"],
        env=eval_env,
        gamma=config["teacher_gamma"],
        verbose=config["verbose"],
        student_gamma=config["student_gamma"],
        buffer_size=config["buffer_size"],
        ent_coef=config["ent_coef"],
        student_ent_coef=config["student_ent_coef"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        gradient_steps=config["gradient_steps"],
        expert_replaybuffer=config["expert_replaybuffer"],
        expert_replaybuffersize=config["expert_replaybuffersize"],
        # tensorboard_log=f"logs/tensorboard/{run.name}/",
        seed=config["seed"],
        learning_starts=0,
        student_begin=config["student_begin"],
        reward_reg_param=config["reward_reg_param"],
        student_domain_randomization_scale=config["student_domain_randomization_scale"],
        explorer=config["explorer"],
        teacher_state_only_reward=config["state_only"],
    )
    irl_model.set_parameters(
        "/home/feiyang/Repositories/RoboticsGym/logs/ICML2024 Guided Learning/Student imitating POMDP 0.1 w CL/student/best_model.zip"
    )

    # Evaluation
    mean_student_reward, mean_student_len = evaluate_student_policy(
        irl_model, eval_env, n_eval_episodes=1
    )
    print(
        f"Average student reward: {mean_student_reward}, with length {mean_student_len}"
    )

    return mean_student_reward


def visualize_best_student():
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 5e6,
        "env_id": "CassieMirror-v1",
        "buffer_size": 1000000,
        "train_freq": 3,
        "gradient_steps": 3,
        "progress_bar": True,
        "verbose": 1,
        "ent_coef": "auto",
        "student_ent_coef": "auto",
        "learning_rate": linear_schedule(5e-3),
        "n_envs": 24,
        "batch_size": 300,
        "seed": 1,
        "expert_replaybuffersize": 600,
        "expert_replaybuffer": "expert_demo/SAC/10traj_morestable",
    }

    # Create log dir
    train_env = make_vec_env(
        config["env_id"], n_envs=config["n_envs"], vec_env_cls=SubprocVecEnv
    )
    # Separate evaluation env
    eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render": True},
    )

    # Init model
    irl_model = HIP.load("logs/2023-07-07-11-16-17/student/best_model.zip")
    irl_model.set_parameters("logs/2023-07-08-21-42-09/student/best_model.zip")

    # Evaluation
    mean_student_reward, _ = evaluate_student_policy(
        irl_model, eval_env, n_eval_episodes=10
    )

    # Finish wandb run

    return mean_student_reward


def run_mujoco_experiments():
    for env_id in ["Hopper-v4", "Humanoid-v4", "Walker2d-v4"]:
        config = {
            "policy_type": "IPMDPolicy",
            "total_timesteps": 5e6,
            "env_id": env_id,
            "buffer_size": 1000000,
            "train_freq": 3,
            "gradient_steps": 3,
            "progress_bar": True,
            "verbose": 1,
            "ent_coef": "0.001",
            "student_ent_coef": 0.001,
            "learning_rate": 3e-4,
            "n_envs": 12,
            "batch_size": 512,
            "seed": 42,
            "expert_replaybuffersize": 1000,
            "expert_replaybuffer": f"logs/{env_id}/expert_rb",
            "student_begin": int(0),
            "teacher_gamma": 1.00,
            "student_gamma": 1.00,
            "reward_reg_param": 0.05,
        }
        run = wandb.init(
            project="HIP MuJoCo",
            config=config,
            name=f'{env_id}-{time.strftime("%Y-%m-%d-%H-%M-%S")}',
            tags=[f"{env_id}", "MuJoCo Examples"],
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            reinit=True,
            notes="student has fixed ent coef, teacher has auto ent coef",
        )
        wandb.run.log_code(".")

        wandbcallback = WandbCallback(
            gradient_save_freq=5000,
        )
        # Create log dir
        train_env = make_vec_env(
            config["env_id"], n_envs=config["n_envs"], vec_env_cls=SubprocVecEnv
        )
        # Separate evaluation env
        eval_env = make_vec_env(config["env_id"], n_envs=1, vec_env_cls=SubprocVecEnv)
        # Use deterministic actions for evaluation
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"logs/{run.project}/{run.name}/teacher/",
            log_path=f"logs/{run.project}/{run.name}/teacher/",
            eval_freq=2000,
            n_eval_episodes=3,
            deterministic=True,
            render=False,
            verbose=1,
        )
        student_eval_env = make_vec_env(
            config["env_id"], n_envs=1, vec_env_cls=SubprocVecEnv
        )
        eval_student_callback = EvalStudentCallback(
            student_eval_env,
            best_model_save_path=f"logs/{run.project}/{run.name}/student/",
            log_path=f"logs/{run.project}/{run.name}/student/",
            eval_freq=1000,
            n_eval_episodes=3,
            deterministic=True,
            render=False,
            verbose=1,
        )
        callback_list = CallbackList(
            [eval_callback, wandbcallback, eval_student_callback]
        )
        # Init model
        irl_model = HIP(
            policy=config["policy_type"],
            student_policy="IPMDPolicy",
            env=train_env,
            gamma=config["teacher_gamma"],
            verbose=config["verbose"],
            student_gamma=config["student_gamma"],
            buffer_size=config["buffer_size"],
            ent_coef=config["ent_coef"],
            student_ent_coef=config["student_ent_coef"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            gradient_steps=config["gradient_steps"],
            expert_replaybuffer=config["expert_replaybuffer"],
            expert_replaybuffersize=config["expert_replaybuffersize"],
            tensorboard_log=f"logs/tensorboard/{run.name}/",
            seed=config["seed"],
            learning_starts=100,
            student_begin=config["student_begin"],
            reward_reg_param=config["reward_reg_param"],
        )
        # irl_model.set_parameters('/home/feiyang/Develop/Cassie/arm-cassie/logs/2023-07-05-21-23-30/student/best_model.zip')
        # Model learning
        irl_model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback_list,
            progress_bar=config["progress_bar"],
            log_interval=100,
        )
        # Evaluation
        _, _ = evaluate_policy(irl_model, eval_env, n_eval_episodes=10)
        _, _ = evaluate_student_policy(irl_model, eval_env, n_eval_episodes=10)

        # Finish wandb run
        run.finish()


if __name__ == "__main__":
    train_cassie_v4()
