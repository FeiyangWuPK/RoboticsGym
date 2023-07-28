import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from sbx import SAC

if __name__ == '__main__':
    vec_env = make_vec_env('HalfCheetah-v4', n_envs=16, vec_env_cls=SubprocVecEnv)
    eval_vec_env = make_vec_env('HalfCheetah-v4', n_envs=1, vec_env_cls=SubprocVecEnv)
    eval_callback = EvalCallback(eval_vec_env, best_model_save_path="./logs/",
									log_path="./logs/", eval_freq=1000,
									deterministic=True, render=False)
    model = SAC('MlpPolicy', vec_env, verbose=1, )
    # model.set_parameters('logs/best_model.zip')
    model.learn(total_timesteps=3e6, log_interval=10, progress_bar=True, callback=eval_callback)
    mean_reward, mean_len = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward}, Mean len: {mean_len}')

