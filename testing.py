import os
import time
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

# from arm_cassie_env.cassie_env.cassieRLEnvMirror import CassieRLEnvMirror

register(id='Digit-v1',
		entry_point='digit:DigitEnv',
		max_episode_steps=1000,
		autoreset=True,)

register(id='Cassie-v1',
		entry_point='cassie:CassieEnv',
		max_episode_steps=1000,
		autoreset=True,)

register(id='CassieViz-v1',
		entry_point='cassie_viz:CassieEnv',
		max_episode_steps=1000,
		autoreset=True,)

register(id='OldCassie-v1',
		entry_point='oldcassie:OldCassieMirrorEnv',
		max_episode_steps=600,
		autoreset=True,)

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

def visualize_init_stance():
	env = gym.make('CassieViz-v1', render_mode='human')
	evaluate_policy(SAC("MlpPolicy", env), env, render=True, n_eval_episodes=10)

if __name__ == "__main__":
	train = False
	if train:
		# Create the environment
		env = make_vec_env("Cassie-v1", n_envs=32, env_kwargs={'exclude_current_positions_from_observation': False, })
		# Separate evaluation env
		render = False
		if render:
			eval_env = make_vec_env("Cassie-v1", n_envs=1, env_kwargs={'exclude_current_positions_from_observation': False, 'render_mode': 'human'})
		else:
			eval_env = make_vec_env("Cassie-v1", n_envs=1, env_kwargs={'exclude_current_positions_from_observation': False})
		# Use deterministic actions for evaluation
		t = time.strftime("%Y_%m_%d_%H_%M_%S")
		eval_callback = EvalCallback(eval_env, best_model_save_path=f"./logs/{t}/",
									log_path=f"./logs/{t}/", eval_freq=10000,
									deterministic=True, render=render)
		# Init model
		model = SAC("MlpPolicy",
					env,
					verbose=1,
					)
		
		# model.set_parameters("./logs/2023_06_27_13_52_12/best_model")

		# Train the agent
		model.learn(total_timesteps=2e7,
					log_interval=30,
					progress_bar=True,
					callback=eval_callback)
		# eval_env = VecNormalize(make_vec_env("Cassie-v1", n_envs=1, env_kwargs={'exclude_current_positions_from_observation': False, 'render_mode': 'human'}))
		# evaluate_policy(model, eval_env, render=True, n_eval_episodes=10)
	else:
		# load_best_and_visualize()
		# visualize_reference_traj()
		# visualize_init_stance()
		