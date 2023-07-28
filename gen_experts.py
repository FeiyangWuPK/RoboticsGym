import gymnasium as gym
import sys, os
from typing import Callable
import datetime
import time
import optuna
from stable_baselines3.common import type_aliases
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList, EventCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecEnv, sync_envs_normalization, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from newAlgo import HIP, EvalStudentCallback, evaluate_student_policy
from old_cassie.cassie_env.cassieRLEnvMirror import cassieRLEnvMirror
from oldcassie import OldCassieMirrorEnv
from inversepolicies import IPMDPolicy
import wandb
from wandb.integration.sb3 import WandbCallback
from gymnasium.envs.registration import register
import warnings
import numpy as np

def train_SAC(env:str = "HalfCheetah-v4"):
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 5e6,
        "env_id": env,
        'n_envs': 16,
    }
    # Create log dir
    train_env = make_vec_env(config['env_id'], n_envs=config['n_envs'],vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
	# Separate evaluation env
    eval_env = make_vec_env(config['env_id'], n_envs=1,vec_env_cls=SubprocVecEnv,vec_env_kwargs=dict(start_method='fork'))
	# Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=f"./logs/{env}/",
                                 log_path=f"./logs/{env}/", 
                                 eval_freq=2000,
                                 n_eval_episodes=3,
                                 deterministic=True, 
                                 render=False,
                                 verbose=1)
    
    callback_list = CallbackList([eval_callback, ])
	# Init model
    sac_model = SAC(policy=config['policy_type'],
                    env=train_env,
                    tensorboard_log=f'logs/tensorboard/',
                    
                    )
    sac_model.learn(total_timesteps=int(config['total_timesteps']), callback=callback_list, progress_bar=True)

def gen_expert_rb(env_id:str):
    env = make_vec_env(env_id, n_envs=1,vec_env_cls=SubprocVecEnv,vec_env_kwargs=dict(start_method='fork'))
    model = SAC(policy="MlpPolicy", env=env, verbose=1, train_freq=1000 * 10)
    model.set_parameters(f"logs/{env_id}/best_model.zip")
    _, callback = model._setup_learn(1000 * 10, callback=None, )
    model.collect_rollouts(model.env,
                train_freq=model.train_freq,
                action_noise=model.action_noise,
                callback=callback,
                learning_starts=0,
                replay_buffer=model.replay_buffer,
                log_interval=1,)
    print(f' {env_id}: {model.replay_buffer.rewards.sum()/10}')
    model.save_replay_buffer(f"logs/{env_id}/expert_rb")

for env in [ 'HalfCheetah-v4', 'Hopper-v4', 'Humanoid-v4', 'Walker2d-v4']:
    gen_expert_rb(env_id=env)