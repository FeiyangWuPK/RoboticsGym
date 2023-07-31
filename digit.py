import os
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import mujoco

from reference_trajectories.loadDigit import DigitTrajectory

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


class DigitEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 400,
    }

    def __init__(
            self,
            forward_reward_weight=1.25,
            ctrl_cost_weight=0.1,
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.8, 2.0),
            reset_noise_scale=1e-3,
            exclude_current_positions_from_observation=True,
            **kwargs,
    ):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(1003,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(1005,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            os.getcwd()+"/digit-v3.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        # print(self.action_space.shape)
        self.ref_trajectory = DigitTrajectory("reference_trajectories/digit_state_downsample.csv")
        
        initial_qpos, initial_qvel = self.ref_trajectory.state(0)

        self.init_qpos = initial_qpos
        self.init_qvel = initial_qvel
        
        # Index from README. The toes are actuated by motor A and B.
        self.p_index = [7, 8, 9, 14, 18, 23, 30, 31, 32, 33, 
                        34, 35, 36, 41, 45, 50, 57, 58, 59, 60]
        self.v_index = [6, 7, 8, 12, 16, 20, 26, 27, 28, 29, 
                        30, 31, 32, 36, 40, 44, 50, 51, 52, 53]
        
        self.reset_model()

    @property
    def healthy_reward(self):
        return (
                float(self.is_healthy or self._terminate_when_unhealthy)
                * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def _step_mujoco_simulation(self, ctrl, n_frames):
        # Set the control target, this userdata is used by PD control callback.
        self.data.userdata[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)
        
    def step(self, action):
        ref_qpos, ref_qvel = self.ref_trajectory.state(self.timestamp)
        self.timestamp += 1
        
        xy_position_before = mass_center(self.model, self.data)
        
        # self.do_simulation(np.zeros(20), self.frame_skip)
        q_pos_modified = action + ref_qpos[self.p_index]
        self._step_mujoco_simulation(q_pos_modified, self.frame_skip)
        
        position = self.data.qpos.flat.copy()
        rod_index = [10,11,12,13, 19,20,21,22, 24,25,26,27, 
                     37,38,39,40, 46,47,48,49, 51,52,53,54]
        ref_qpos[rod_index] = position[rod_index]
        self.set_state(ref_qpos, ref_qvel)
        
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - ctrl_cost
        terminated = self.terminated
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        self.timestamp = 0
        observation = self._get_obs()
        
        self.data.userdata = np.zeros(20)  # Use userdata as target position.
        # Define a callback that modify the ctrl before mj_step.

        mujoco.set_mjcb_control(None)
        mujoco.set_mjcb_control(lambda m, d: PD_control_CB(m, d))
        
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

def PD_control_CB(model, data):
    kp = np.array([100, 100, 88, 96, 50, 50, 50, 50, 50, 50, 100, 100, 88, 96, 50, 50, 50, 50, 50, 50])
    kd = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    p_index = [7, 8, 9, 14, 18, 23, 30, 31, 32, 33, 34, 35, 36, 41, 45, 50, 57, 58, 59, 60]
    v_index = [6, 7, 8, 12, 16, 20, 26, 27, 28, 29, 30, 31, 32, 36, 40, 44, 50, 51, 52, 53]
    p = data.qpos[p_index]
    v = data.qvel[v_index]
    p_desired = data.userdata[:]
    v_desired = np.zeros(20)
    data.ctrl[:] = kp * (p_desired - p) + kd * (v_desired - v)