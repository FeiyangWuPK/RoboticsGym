import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback

register(id='Digit-v1',
		entry_point='digit:DigitEnv',
		max_episode_steps=600,
		autoreset=True,)

register(id='Cassie-v1',
		entry_point='cassie:CassieEnv',
		max_episode_steps=600,
		autoreset=True,)

register(id='CassieViz-v1',
		entry_point='cassie_viz:CassieEnv',
		max_episode_steps=600,
		autoreset=True,)

from typing import Callable


def load_best_and_visualize():
	env = make_vec_env("Cassie-v1", n_envs=1, env_kwargs={'exclude_current_positions_from_observation': False, 'render_mode': 'human'})
	model = SAC("MlpPolicy",
				env,
				verbose=0,
				learning_rate=1e-3,)
	model.set_parameters("./logs/best_model.zip")
	evaluate_policy(model, env, render=True, n_eval_episodes=10)
	
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
		"total_timesteps": int(1e7),
		"env_id": "Cassie-v1",
		"progress_bar": True,
		"verbose": 1,
		"learning_rate": 3e-4,
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
	env = make_vec_env(config['env_id'], n_envs=4,)
	# Separate evaluation env
	eval_env = make_vec_env(config['env_id'], n_envs=1,)
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
	train = True
	if train:
		train_model()
	else:
		visualize_reference_traj()