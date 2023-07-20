import gymnasium as gym
import sys
from typing import Callable
import datetime
import time
import optuna
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac import SAC
from stable_baselines3.common.env_checker import check_env
from ipmd import IPMD
from arm_cassie_env.cassie_env.cassieRLEnvMirror import cassieRLEnvMirror
from arm_cassie_env.cassie_env.oldCassie import OldCassieMirrorEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from gymnasium.envs.registration import register

register(
        id='CassieMirror-v1',
        entry_point='arm_cassie_env.cassie_env.oldCassie:OldCassieMirrorEnv',
        max_episode_steps=600,
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

def make_env(env_id):
    def _f():
        if env_id == 0:
            env = cassieRLEnvMirror(visual=True)
        else:
            env = cassieRLEnvMirror(visual=False)
        return env
    return _f

def train_expert_policy():
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1e7,
        "env_id": "CassieMirror-v1",
        'buffer_size': 1000000,
        'train_freq': 5,
        'gradient_steps': 1,
        "progress_bar": True,
        "verbose": 0,
        'ent_coef': 'auto',
        'student_ent_coef': 0.01,
        'learning_rate': linear_schedule(5e-3),
        "n_envs": 64,
        'batch_size': 256,
        'seed': 1,
        'expert_replaybuffersize': 600,
        'expert_replaybuffer': 'expert_demo/SAC/buffer10traj',
    }
    run = wandb.init(
        project="CassieExpertGen",
        config=config,
        name=f'{time.strftime("%Y-%m-%d-%H-%M-%S")}',
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    wandbcallback = WandbCallback(
            model_save_path=f"models/{run.id}",
            model_save_freq=2000,
            gradient_save_freq=2000,
            verbose=0,
        )
    # Create log dir
    train_env = make_vec_env(config['env_id'], 
                             n_envs=config['n_envs'], 
                             vec_env_cls=SubprocVecEnv)
	# Separate evaluation env
    eval_env = make_vec_env(config['env_id'], 
                            n_envs=1, 
                            vec_env_cls=SubprocVecEnv)
	# Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=f"./logs/{run.name}/",
                                 log_path=f"./logs/{run.name}/", 
                                 eval_freq=2000,
                                 n_eval_episodes=5,
                                 deterministic=True, 
                                 render=False)
    callback_list = CallbackList([eval_callback, wandbcallback])
	# Init model
    irl_model = SAC('MlpPolicy', 
                     env=train_env, 
                     gamma=0.99, 
                     verbose=config['verbose'],
                     buffer_size=config['buffer_size'],
                     ent_coef=config['ent_coef'], 
                     batch_size=config['batch_size'], 
                     learning_rate=config['learning_rate'],
                     gradient_steps=config['gradient_steps'],
                     learning_starts=100,
                     tensorboard_log=f'logs/tensorboard/{run.name}/'
                     )
    # Model learning
    irl_model.learn(
        total_timesteps=config['total_timesteps'], 
        callback=callback_list,
        progress_bar=config['progress_bar'],
        )
    return f'logs/{run.name}/best_model'

def obtain_expert_traj(model_path:str, num_traj:int):
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 3e6,
        "env_id": "CassieMirror-v1",
        "progress_bar": True,
        "verbose": 1,
        'ent_coef': 0.01,
        'learning_rate': 5e-3,
        "n_envs": 32,
        'gradient_steps': 1, 
        'batch_size': 512,
        'buffer_size': 200000,
    }
    eval_env = make_vec_env(config['env_id'], n_envs=1,vec_env_cls=SubprocVecEnv)
    model = SAC('MlpPolicy', env = eval_env, train_freq=600 * num_traj)
    model.set_parameters(model_path)
    _, callback = model._setup_learn(600 * num_traj, callback=None, )
    model.collect_rollouts(model.env,
                train_freq=model.train_freq,
                action_noise=model.action_noise,
                callback=callback,
                learning_starts=0,
                replay_buffer=model.replay_buffer,
                log_interval=1,)
    
    model.save_replay_buffer('expert_demo/SAC/10traj_morestable')
    print(model.replay_buffer.rewards.sum())

def visualize_expert_agent_traj(model_path:str):
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 3e6,
        "env_id": "CassieMirror-v1",
        "progress_bar": True,
        "verbose": 1,
        'ent_coef': 0.01,
        'learning_rate': 5e-3,
        "n_envs": 32,
        'gradient_steps': 1, 
        'batch_size': 512,
        'buffer_size': 200000,
    }
    eval_env = make_vec_env(config['env_id'], 
                            n_envs=1, 
                            vec_env_cls=SubprocVecEnv,
                            env_kwargs={'render': True})
    
    model = SAC.load(model_path, env=eval_env)
    mean_reward, _ = evaluate_policy(model, n_eval_episodes=10, env=eval_env, render=False)
    print(f'Mean Reward for RL expert agent: {mean_reward}')

def train_ipmd_agent():
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1e7,
        "env_id": "CassieMirror-v1",
        "progress_bar": True,
        "verbose": 0,
        'ent_coef': 'auto',
        'learning_rate': linear_schedule(3e-4),
        'train_freq': 1,
        "n_envs": 72,
        'gradient_steps': 1, 
        'batch_size': 256,
        'buffer_size': int(1e6),
        'expert_replay_buffer_loc': 'expert_demo/SAC/10traj_morestable',
        'expert_traj_size': 600,
    }
    run = wandb.init(
        project="IRL IPMD Param Optimization",
        config=config,
        name=f'{time.strftime("%Y-%m-%d-%H-%M-%S")}',
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    wandb.run.log_code(".")
    wandbcallback = WandbCallback(
            verbose=2,
        )
    # Create log dir
    train_env = make_vec_env(config['env_id'], n_envs=config['n_envs'], vec_env_cls=SubprocVecEnv)
	# Separate evaluation env
    eval_env = make_vec_env(config['env_id'], n_envs=1, vec_env_cls=SubprocVecEnv)
	# Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=f"./logs/{run.name}/",
                                 log_path=f"./logs/{run.name}/", 
                                 eval_freq=5000,
                                 n_eval_episodes=5,
                                 deterministic=True, 
                                 render=False)
    callback_list = CallbackList([eval_callback, wandbcallback])
	# Init model
    irl_model = IPMD('MlpPolicy', 
                     env=train_env, 
                     gamma=0.99, 
                     verbose=config['verbose'],
                     buffer_size=config['buffer_size'],
                     ent_coef=config['ent_coef'], 
                     batch_size=config['batch_size'], 
                     learning_rate=config['learning_rate'],
                     gradient_steps=config['gradient_steps'],
                     learning_starts=100,
                     expert_replay_buffer_loc=config['expert_replay_buffer_loc'], 
                     expert_traj_size=config['expert_traj_size'],
                     tensorboard_log=f'logs/tensorboard/{run.name}/'
                     )
    # Model learning
    irl_model.learn(
        total_timesteps=config['total_timesteps'], 
        callback=callback_list,
        progress_bar=config['progress_bar'],
        log_interval=1,
        )
    # Evaluation
    mean_reward, _ = evaluate_policy(irl_model, n_eval_episodes=10, env=eval_env)
    # Finish wandb run
    run.finish()

    return mean_reward

def visualize_irl_agent_traj(model_path:str):
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 3e6,
        "env_id": "CassieMirror-v1",
        "progress_bar": True,
        "verbose": 1,
        'ent_coef': 0.01,
        'learning_rate': 5e-3,
        "n_envs": 32,
        'gradient_steps': 1, 
        'batch_size': 512,
        'buffer_size': 200000,
    }
    eval_env = make_vec_env(config['env_id'], 
                            n_envs=1, 
                            vec_env_cls=SubprocVecEnv,
                            env_kwargs={'render': True})
    
    model = IPMD.load(model_path, env=eval_env)
    mean_reward, _ = evaluate_policy(model, n_eval_episodes=10, env=eval_env, render=False)
    

if __name__ == '__main__':
    # best_model_path = train_expert_policy()
    # visualize_expert_agent_traj(model_path=best_model_path)
    # visualize_expert_agent_traj('logs/2023-07-11-13-24-37/best_model.zip')
    # obtain_expert_traj('logs/2023-07-11-13-24-37/best_model.zip', 10)
    train_ipmd_agent()
    # visualize_irl_agent_traj('logs/2023-07-12-07-54-02-Entropy0.01/best_model.zip')