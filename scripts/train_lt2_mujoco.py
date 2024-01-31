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

from roboticsgym.algorithms.sb3.newAlgo_student_only import HIPSTUDENTONLY


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


def test_student_policy():
    config = {
        "policy_type": "IPMDPolicy",
        "total_timesteps": 5e6,
        "env_id": "CassieMirror-v5",
        "buffer_size": 200000,
        "train_freq": 5,
        "gradient_steps": 5,
        "progress_bar": True,
        "verbose": 0,
        "ent_coef": "auto",
        "student_ent_coef": "auto",
        "learning_rate": "1e-3",
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


def run_mujoco_rl(env_name):
    config = {
        "teacher_policy_type": "IPMDPolicy",
        "student_policy_type": "IPMDPolicy",
        "total_timesteps": 1e6,
        "env_id": "NoisyMujoco-v4",
        "buffer_size": int(1e6),
        "train_freq": 1,
        "gradient_steps": 1,
        "progress_bar": True,
        "verbose": 0,
        "ent_coef": "auto",
        "student_ent_coef": "auto",
        "learning_rate": 3e-4,
        "n_envs": 8,
        "batch_size": 256,
        "seed": 1,
        "expert_replaybuffersize": 600,
        "expert_replaybuffer": "expert_trajectories/cassie_v4/10traj_morestable.pkl",
        "student_begin": int(0),
        "teacher_gamma": 0.99,
        "student_gamma": 0.99,
        "reward_reg_param": 0.05,
        "student_domain_randomization_scale": 0.4,
        "explorer": "teacher",
        "state_only": False,
        "testing_pomdp": False,
        "thv_imitation_learning": True,
    }
    run = wandb.init(
        project="ICML2024 Guided Learning MuJoCo RL",
        config=config,
        # name=config["env_id"] + f'-{time.strftime("%Y-%m-%d-%H-%M-%S")}',
        name=f"Mujoco RL {env_name} {config['student_domain_randomization_scale']} Jan 30 seed 1",
        tags=[config["env_id"]],
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        reinit=True,
        notes="",
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
            "task": env_name,
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )
    # Separate evaluation env
    teacher_eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "task": env_name,
        },
    )
    # Use deterministic actions for evaluation
    teacher_eval_callback = EvalTeacherCallback(
        teacher_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/teacher/",
        log_path=f"logs/{run.project}/{run.name}/teacher/",
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    student_eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "task": env_name,
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )
    eval_student_callback = EvalStudentCallback(
        student_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/student/",
        log_path=f"logs/{run.project}/{run.name}/student/",
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callback_list = CallbackList(
        [teacher_eval_callback, wandbcallback, eval_student_callback]
    )
    # Init model
    irl_model = HIP(
        policy=config["teacher_policy_type"],
        student_policy=config["student_policy_type"],
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
        testing_pomdp=config["testing_pomdp"],
        thv_imitation_learning=config["thv_imitation_learning"],
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


def run_mujoco_second_stage(env_name):
    config = {
        "teacher_policy_type": "IPMDPolicy",
        "student_policy_type": "IPMDPolicy",
        "total_timesteps": 1e6,
        "env_id": "NoisyMujoco-v4",
        "buffer_size": int(1e6),
        "train_freq": 1,
        "gradient_steps": 1,
        "progress_bar": True,
        "verbose": 0,
        "ent_coef": "auto",
        "student_ent_coef": "auto",
        "learning_rate": 3e-4,
        "n_envs": 8,
        "batch_size": 256,
        "seed": 42,
        "student_begin": int(0),
        "teacher_gamma": 0.99,
        "student_gamma": 0.99,
        "student_domain_randomization_scale": 0.4,
        "explorer": "student",
        "state_only": False,
        "testing_pomdp": False,
        "thv_imitation_learning": True,
        "teacher_policy_path": f"/home/feiyang/Documents/Repositories/RoboticsGym/logs/ICML2024 Guided Learning MuJoCo RL/Mujoco RL {env_name} 0.4/teacher/best_model.zip",
    }
    run = wandb.init(
        project="ICML2024 Guided Learning MuJoCo RL",
        config=config,
        # name=config["env_id"] + f'-{time.strftime("%Y-%m-%d-%H-%M-%S")}',
        name=f"Mujoco RL {env_name} {config['student_domain_randomization_scale']} Student Learns from BC",
        tags=[config["env_id"]],
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        reinit=True,
        notes="",
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
            "task": env_name,
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )
    # Separate evaluation env
    teacher_eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "task": env_name,
        },
    )
    # Use deterministic actions for evaluation
    teacher_eval_callback = EvalTeacherCallback(
        teacher_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/teacher/",
        log_path=f"logs/{run.project}/{run.name}/teacher/",
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    student_eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "task": env_name,
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )
    eval_student_callback = EvalStudentCallback(
        student_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/student/",
        log_path=f"logs/{run.project}/{run.name}/student/",
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callback_list = CallbackList(
        [teacher_eval_callback, wandbcallback, eval_student_callback]
    )
    # Init model
    irl_model = HIPSTUDENTONLY(
        policy=config["teacher_policy_type"],
        student_policy=config["student_policy_type"],
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
        tensorboard_log=f"logs/tensorboard/{run.name}/",
        seed=config["seed"],
        learning_starts=100,
        student_begin=config["student_begin"],
        student_domain_randomization_scale=config["student_domain_randomization_scale"],
        explorer=config["explorer"],
        teacher_state_only_reward=config["state_only"],
        testing_pomdp=config["testing_pomdp"],
        thv_imitation_learning=config["thv_imitation_learning"],
        teacher_policy_path=config["teacher_policy_path"],
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


def train_hopper_rl():
    env_name = "Hopper-v4"
    config = {
        "teacher_policy_type": "IPMDPolicy",
        "student_policy_type": "IPMDPolicy",
        "total_timesteps": 1e6,
        "env_id": "NoisyMujoco-v4",
        "buffer_size": int(1e6),
        "train_freq": 1,
        "gradient_steps": 1,
        "progress_bar": True,
        "verbose": 0,
        "ent_coef": "auto",
        "student_ent_coef": "auto",
        "learning_rate": linear_schedule(3e-3),
        "batch_size": 256,
        "seed": 42,
        "expert_replaybuffersize": 600,
        "expert_replaybuffer": "expert_trajectories/cassie_v4/10traj_morestable.pkl",
        "student_begin": int(0),
        "teacher_gamma": 0.99,
        "student_gamma": 0.99,
        "reward_reg_param": 0.05,
        "student_domain_randomization_scale": 0.2,
        "explorer": "teacher",
        "n_envs": 8,
        "state_only": False,
        "testing_pomdp": False,
        "thv_imitation_learning": True,
    }
    run = wandb.init(
        project="ICML2024 Guided Learning MuJoCo RL",
        config=config,
        name=f"Mujoco RL {env_name} {config['student_domain_randomization_scale']}",
        tags=[config["env_id"]],
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        reinit=True,
        notes="combine asym and bc loss",
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
            "task": env_name,
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )
    # Separate evaluation env
    teacher_eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "task": env_name,
        },
    )
    # Use deterministic actions for evaluation
    teacher_eval_callback = EvalTeacherCallback(
        teacher_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/teacher/",
        log_path=f"logs/{run.project}/{run.name}/teacher/",
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=False,
        render=False,
        verbose=1,
    )
    student_eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "task": env_name,
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )
    eval_student_callback = EvalStudentCallback(
        student_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/student/",
        log_path=f"logs/{run.project}/{run.name}/student/",
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=False,
        render=False,
        verbose=1,
    )
    callback_list = CallbackList(
        [teacher_eval_callback, wandbcallback, eval_student_callback]
    )
    # Init model
    irl_model = HIP(
        policy=config["teacher_policy_type"],
        student_policy=config["student_policy_type"],
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
        testing_pomdp=config["testing_pomdp"],
        thv_imitation_learning=config["thv_imitation_learning"],
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

def run_hopper_second_stage(env_name="Hopper-v4"):
    config = {
        "teacher_policy_type": "IPMDPolicy",
        "student_policy_type": "IPMDPolicy",
        "total_timesteps": 1e6,
        "env_id": "NoisyMujoco-v4",
        "buffer_size": int(1e6),
        "train_freq": 1,
        "gradient_steps": 1,
        "progress_bar": True,
        "verbose": 0,
        "ent_coef": "auto",
        "student_ent_coef": "auto",
        "learning_rate": 3e-4,
        "n_envs": 10,
        "batch_size": 256,
        "seed": 42,
        "student_begin": int(0),
        "teacher_gamma": 0.99,
        "student_gamma": 0.99,
        "student_domain_randomization_scale": 0.2,
        "explorer": "student",
        "state_only": False,
        "testing_pomdp": False,
        "thv_imitation_learning": True,
        "teacher_policy_path": f"/home/feiyang/Documents/Repositories/RoboticsGym/logs/ICML2024 Guided Learning MuJoCo RL/Mujoco RL {env_name} 0.2/teacher/best_model.zip",
    }
    run = wandb.init(
        project="ICML2024 Guided Learning MuJoCo RL",
        config=config,
        # name=config["env_id"] + f'-{time.strftime("%Y-%m-%d-%H-%M-%S")}',
        name=f"Mujoco RL {env_name} {config['student_domain_randomization_scale']} Student Learns from BC",
        tags=[config["env_id"]],
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        reinit=True,
        notes="",
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
            "task": env_name,
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )
    # Separate evaluation env
    teacher_eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "task": env_name,
        },
    )
    # Use deterministic actions for evaluation
    teacher_eval_callback = EvalTeacherCallback(
        teacher_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/teacher/",
        log_path=f"logs/{run.project}/{run.name}/teacher/",
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=False,
        render=False,
        verbose=1,
    )
    student_eval_env = make_vec_env(
        config["env_id"],
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "task": env_name,
            "domain_randomization_scale": config["student_domain_randomization_scale"],
        },
    )
    eval_student_callback = EvalStudentCallback(
        student_eval_env,
        best_model_save_path=f"logs/{run.project}/{run.name}/student/",
        log_path=f"logs/{run.project}/{run.name}/student/",
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=False,
        render=False,
        verbose=1,
    )
    callback_list = CallbackList(
        [teacher_eval_callback, wandbcallback, eval_student_callback]
    )
    # Init model
    irl_model = HIPSTUDENTONLY(
        policy=config["teacher_policy_type"],
        student_policy=config["student_policy_type"],
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
        tensorboard_log=f"logs/tensorboard/{run.name}/",
        seed=config["seed"],
        learning_starts=100,
        student_begin=config["student_begin"],
        student_domain_randomization_scale=config["student_domain_randomization_scale"],
        explorer=config["explorer"],
        teacher_state_only_reward=config["state_only"],
        testing_pomdp=config["testing_pomdp"],
        thv_imitation_learning=config["thv_imitation_learning"],
        teacher_policy_path=config["teacher_policy_path"],
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