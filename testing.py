import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback

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

if __name__ == "__main__":
	train = True
	if train:
		# Create the environment
		env = make_vec_env("Cassie-v1", n_envs=16, env_kwargs={'exclude_current_positions_from_observation': False})
		# Separate evaluation env
		eval_env = make_vec_env("Cassie-v1", n_envs=1, env_kwargs={'exclude_current_positions_from_observation': False})
		# Use deterministic actions for evaluation
		eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
									log_path="./logs/", eval_freq=1000,
									deterministic=True, render=False)
		# Init model
		model = SAC("MlpPolicy",
					env,
					buffer_size=200000,
					verbose=1,
					ent_coef=0.01,
					learning_rate=5e-3,)

		# Train the agent
		model.learn(total_timesteps=6e6,
					log_interval=100,
					progress_bar=True,
					callback=eval_callback)
	else:
		visualize_reference_traj()