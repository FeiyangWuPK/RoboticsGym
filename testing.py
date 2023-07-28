import os
import time
from typing import Callable

from stable_baselines3 import SAC
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


# register(id='Digit-v1',
# 		entry_point='digit:DigitEnv',
# 		max_episode_steps=1000,
# 		autoreset=True,)

register(id='Cassie-v1',
		entry_point='cassie:CassieEnv',
		max_episode_steps=600,
		autoreset=True,)

register(id='MjCassie-v2',
		entry_point='mj_cassie:CassieEnv',
		max_episode_steps=600,
		autoreset=True,)

register(id='CassieViz-v1',
		entry_point='cassie_viz:CassieEnv',
		max_episode_steps=1000,
		autoreset=True,)

# register(id='OldCassie-v1',
# 		entry_point='oldcassie:OldCassieMirrorEnv',
# 		max_episode_steps=600,
# 		autoreset=True,)

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

def load_best_and_visualize():
	env = make_vec_env("MjCassie-v2", n_envs=1, env_kwargs={'exclude_current_positions_from_observation': False, 'render_mode': 'human'})
	best_irl_model = SAC("MlpPolicy",
				env,
				verbose=1,
				learning_rate=1e-3,
				train_freq=600)
	best_irl_model.set_parameters("logs/best_model.zip")
	_, callback = best_irl_model._setup_learn(600, callback=None, )
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
		"total_timesteps": int(1e7),
		"env_id": "MjCassie-v2",
		"progress_bar": True,
		"verbose": 0,
		"learning_rate": linear_schedule(5e-3),
		"n_envs": 1,
	}
	run = wandb.init(
		project="New cassie env",
		config=config,
		name=f'{time.strftime("%m%d%H%M")}-PDController',
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		# monitor_gym=True,  # auto-upload the videos of agents playing the game
		notes="with PD controller, and with additional observational space",
		save_code=True,  # optional
	)
	wandb.run.log_code(".")
	wandbcallback = WandbCallback(
			# model_save_path=f"models/{run.id}",
			# model_save_freq=2000,
			# gradient_save_freq=2000,
			verbose=2,
		)
	env = make_vec_env(config['env_id'], n_envs=config['n_envs'], vec_env_cls=SubprocVecEnv, env_kwargs={'exclude_current_positions_from_observation': False, 
		 'render_mode':'human'})
	# Separate evaluation env
	eval_env = make_vec_env(config['env_id'], n_envs=1, vec_env_cls=SubprocVecEnv, env_kwargs={'exclude_current_positions_from_observation': False, })
	# Use deterministic actions for evaluation
	eval_callback = EvalCallback(eval_env, best_model_save_path=f"logs/{run.name}/",
									log_path=f"logs/{run.name}/", eval_freq=5000,
									deterministic=True, render=False)
	callback_list = CallbackList([eval_callback, wandbcallback])
	# Init model
	model = SAC("MlpPolicy",
				env,
				verbose=config["verbose"],
				# ent_coef=0.01,
				learning_rate=config['learning_rate'],
				tensorboard_log=f'logs/tensorboard/{run.name}/',)
	
	model.learn(
		total_timesteps=config["total_timesteps"],
		callback=callback_list,
		progress_bar=config["progress_bar"],
		# log_interval=100,
	)
	run.finish()

if __name__ == "__main__":
	# train_model()
	# load_best_and_visualize()
	visualize_reference_traj()
		