from old_cassie.cassie_env.cassieRLEnvMirror import cassieRLEnvMirror
from old_cassie.cassie_env.quaternion_function import quat2yaw
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
from gymnasium import utils


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

# a class that has all the CassieMirrorEnv functions but does not inherites from all the other classes
class OldCassieMirrorEnv(gym.Env, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
        }
    def __init__(
            self,
            forward_reward_weight=1.25,
            ctrl_cost_weight=0.1,
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.8, 2.0),
            reset_noise_scale=1e-3,
            **kwargs,
        ):
        self.env = cassieRLEnvMirror()
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale
        
        self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(self._get_obs().shape[0],), dtype=np.float64
            )
        
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)

    def reset(self, seed=None, options=None):
        self.env.reset()
        return self.env.get_state(), {}
    
    def _get_obs(self):
        return self.env.get_state()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info 
    
    # def viewer_setup(self):
    #     assert self.viewer is not None
    #     for key, value in DEFAULT_CAMERA_CONFIG.items():
    #         if isinstance(value, np.ndarray):
    #             getattr(self.viewer.cam, key)[:] = value
    #         else:
    #             setattr(self.viewer.cam, key, value)