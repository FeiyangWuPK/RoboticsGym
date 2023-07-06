import os
import time
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback

# from arm_cassie_env.cassie_env.cassieRLEnvMirror import CassieRLEnvMirror

register(id='Digit-v1',
		entry_point='digit:DigitEnv',
		max_episode_steps=1000,
		autoreset=True,)

register(id='Cassie-v1',
		entry_point='cassie:CassieEnv',
		max_episode_steps=1000,
		autoreset=True,)

register(id='MjCassie-v1',
		entry_point='mj_cassie:CassieEnv',
		max_episode_steps=600,
		autoreset=True,)

register(id='CassieViz-v1',
		entry_point='cassie_viz:CassieEnv',
		max_episode_steps=1000,
		autoreset=True,)

register(id='OldCassie-v1',
		entry_point='oldcassie:OldCassieMirrorEnv',
		max_episode_steps=600,
		autoreset=True,)

env = gym.make('OldCassie-v1')
check_env(env, warn=True)

from typing import Callable


def load_best_and_visualize():
	env = VecNormalize(make_vec_env("Cassie-v1", n_envs=1, env_kwargs={'exclude_current_positions_from_observation': False, 'render_mode': 'human'}))
	best_irl_model = SAC("MlpPolicy",
				env,
				verbose=1,
				learning_rate=1e-3,
				train_freq=1000)
	best_irl_model.set_parameters("/home/feiyang/Develop/Cassie/arm-cassie/arm_cassie_env/logs/SAC/2023_06_26_23_53_46/best_model/best_model.zip")
	_, callback = best_irl_model._setup_learn(100000, callback=None, )
	best_irl_model.collect_rollouts(best_irl_model.env,
                train_freq=best_irl_model.train_freq,
                action_noise=best_irl_model.action_noise,
                callback=callback,
                learning_starts=0,
                replay_buffer=best_irl_model.replay_buffer,
                log_interval=1,)
	
	
def visualize_reference_traj():
	env = make_vec_env("CassieViz-v1", n_envs=1, env_kwargs={'exclude_current_positions_from_observation': False, 'render_mode': 'human'})
	model = SAC("MlpPolicy",
				env,
				verbose=0,
				learning_rate=1e-3,)
	evaluate_policy(model, env, render=True, n_eval_episodes=10)

def train_model():
	config = {
		"policy_type": "MlpPolicy",
		"total_timesteps": int(3e6),
		"env_id": "MjCassie-v1",
		"progress_bar": True,
		"verbose": 1,
		"learning_rate": 5e-3,
		"n_envs": 16,
	}
	run = wandb.init(
		project="sb3",
		config=config,
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		monitor_gym=True,  # auto-upload the videos of agents playing the game
		save_code=True,  # optional
	)
	wandbcallback = WandbCallback(
			model_save_path=f"models/{run.id}",
			model_save_freq=10000,
			gradient_save_freq=10000,
			verbose=2,
		)
	env = make_vec_env(config['env_id'], n_envs=config['n_envs'], vec_env_cls=SubprocVecEnv, env_kwargs={'exclude_current_positions_from_observation': True})
	# Separate evaluation env
	eval_env = make_vec_env(config['env_id'], n_envs=1, vec_env_cls=SubprocVecEnv, env_kwargs={'exclude_current_positions_from_observation': True, 'render_mode': 'human'})
	# Use deterministic actions for evaluation
	eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
									log_path="./logs/", eval_freq=10000,
									deterministic=True, render=False)
	callback_list = CallbackList([eval_callback, wandbcallback])
	# Init model
	model = SAC("MlpPolicy",
				env,
				verbose=config["verbose"],
				learning_rate=config['learning_rate'],)
	
	model.learn(
		total_timesteps=config["total_timesteps"],
		callback=callback_list,
		progress_bar=config["progress_bar"],
		log_interval=100,
	)
	run.finish()

if __name__ == "__main__":
	train_model()
		