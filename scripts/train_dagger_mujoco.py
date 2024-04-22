import numpy as np
import imageio
from datetime import datetime
from tqdm import tqdm

import logging

import torch

from imitation.policies.serialize import load_policy
from imitation.util import util

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from roboticsgym.algorithms.dagger_imitation import BC, SimpleDAggerTrainer

from roboticsgym.envs.noisy_mujoco import NoisyMujocoEnv 


from torch.utils.tensorboard import SummaryWriter



def train_dagger(env_name, n_envs, total_steps):
    env = util.make_vec_env(
        env_name,
        rng=np.random.default_rng(),
        n_envs=n_envs,
        env_make_kwargs={"render_mode": "rgb_array"},
    )

    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)

    

    obs = env.reset()

    expert = load_policy(
        "sac-huggingface",
        organization="sb3",
        env_name="HalfCheetah-v3",
        venv=env,
    )

    train_env = make_vec_env(
            "NoisyMujoco-v4",
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={
                "task": env_name,
                "domain_randomization_scale": 0.1,
            },
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    
    
    tb_writer = SummaryWriter(log_dir='tensorboard/')

    bc_trainer = BC(
        observation_space=env.observation_space, 
        action_space=env.action_space, 
        rng=np.random.default_rng(),
        device=device,
        tb_writer=tb_writer)

    dagger_trainer = SimpleDAggerTrainer(
        venv=train_env,
        rng=np.random.default_rng(),
        expert_policy=expert,
        bc_trainer=bc_trainer,
        is_env_noisy=True)
    
    tb_writer.close()


    

    dagger_trainer.train(total_steps)

    # dagger_trainer.save_policy("models/dagger_save")
    # take_video_results(env_name, n_envs,dagger_trainer.policy)


def take_video_results(env_name, n_envs, policy):
    env = util.make_vec_env(
        env_name,
        rng=np.random.default_rng(),
        n_envs=n_envs,
        env_make_kwargs={"render_mode": "rgb_array"},
    )
        
    print("Start video")
    images_trainer = []
    obs = env.reset()
    print(env.render_mode)
    dones = np.zeros(env.num_envs, dtype=bool)
    img = env.render()
    active = np.ones(env.num_envs, dtype=bool)
    # while np.any(active):

    for i in tqdm(range(200)):
        images_trainer.append(img)
        action, _ = policy.predict(obs)
        obs, reward, dones, info = env.step(action)
        img = env.render()

        dones &= active
        active &= ~dones
        
    print(len(images_trainer))

    
    imageio.mimsave(f'videos/dagger_trainer_{datetime.now().strftime("%d_%m_%Y_%H_%M")}.gif', images_trainer)



if __name__ == "__main__":
    train_dagger("HalfCheetah-v4", 4, 10000)
