import os
import time
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
from old_cassie.cassie_env.cassieRLEnvMirror import cassieRLEnvMirror
from ipmd import IPMD
from newAlgo import EvalStudentCallback, evaluate_student_policy, HIP

register(
        id='CassieMirror-v1',
        entry_point='arm_cassie_env.cassie_env.oldCassie:OldCassieMirrorEnv',
        max_episode_steps=600,
        )

def make_env(env_id):
    def _f():
        if env_id == 0:
            env = cassieRLEnvMirror(visual=True)
        else:
            env = cassieRLEnvMirror(visual=False)
        return env
    return _f

from typing import Callable


def load_best_and_visualize_rlexpert():
	env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(1, 33)]))
	best_model = SAC("MlpPolicy",
				env,
				verbose=1,
				learning_rate=1e-3,
				train_freq=6000)
	best_model.set_parameters("logs/autumn-dawn-2/best_model.zip")
	_, callback = best_model._setup_learn(6000, callback=None, )
	best_model.collect_rollouts(best_model.env,
                train_freq=best_model.train_freq,
                action_noise=best_model.action_noise,
                callback=callback,
                learning_starts=0,
                replay_buffer=best_model.replay_buffer,
                log_interval=1,)
	best_model.save_replay_buffer("expert_demo/SAC/teacher_sample_collected")

def load_best_and_visualize_irlagent():
    n_samples = 6000
    train_env = VecMonitor(SubprocVecEnv([make_env(0)]))
    best_model = IPMD('MlpPolicy', env=train_env, 
                     verbose=1, gamma=1.0, 
                     ent_coef=1.0, 
                     batch_size=512, 
                     train_freq=n_samples,
                    #  gradient_steps=5,
                     expert_replay_buffer_loc='expert_demo/SAC/buffer10traj', traj_size=6000)
    # best_model = SAC("MlpPolicy", train_env, verbose=1, learning_rate=1e-3, train_freq=n_samples)
    best_model.set_parameters("/home/feiyang/Develop/Cassie/arm-cassie/arm_cassie_env/logs/ipmd_2023_06_28_23_58_51_ent001_continue_training/best_model.zip")
    # best_model.set_parameters("/home/feiyang/Develop/Cassie/arm-cassie/arm_cassie_env/logs/SAC/2023_06_27_15_13_45/best_model/best_model")
    _, callback = best_model._setup_learn(n_samples, callback=None, )
    best_model.collect_rollouts(best_model.env,
                train_freq=best_model.train_freq,
                action_noise=best_model.action_noise,
                callback=callback,
                learning_starts=0,
                replay_buffer=best_model.replay_buffer,
                log_interval=1,)
    
	# best_model.save_replay_buffer("expert_demo/SAC/buffer10traj")

def load_best_and_visualize_hipagent():
    train_env = make_vec_env('CassieMirror-v1', n_envs=1, seed=0, vec_env_cls=SubprocVecEnv, )
    best_model = HIP.load('logs/2023-07-07-15-51-22/student/best_model.zip', env=train_env)
    evaluate_student_policy(best_model, n_eval_episodes=10, render=True, env=train_env)
        
if __name__ == "__main__":
    # make_env(0)
    # load_best_and_visualize_rlexpert()
    load_best_and_visualize_hipagent()