import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback

register(id='Cassie-v1',
		entry_point='mj_cassie:CassieEnv',
		max_episode_steps=1000,
		autoreset=True,)

if __name__ == "__main__":
	# Create the environment
	env = Monitor(gym.make("Cassie-v1", render_mode="human"))
	# Separate evaluation env
	eval_env = Monitor(gym.make("Cassie-v1"))
	# Use deterministic actions for evaluation
	eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
								log_path="./logs/", eval_freq=500,
								deterministic=True, render=False)
	# Init model
	model = SAC("MlpPolicy",
				env,
				verbose=0,
				learning_rate=1e-3,)

	# Train the agent
	model.learn(total_timesteps=1e6,
				log_interval=5,
				progress_bar=True,
				callback=eval_callback)