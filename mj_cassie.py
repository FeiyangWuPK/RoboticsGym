import os
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import mujoco

from reference_trajectories.loadstep import CassieTrajectory

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


class CassieEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 33,
    }

    def __init__(
            self,
            forward_reward_weight=0.1,
            ctrl_cost_weight=0.1,
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.5, 2.0),
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
                low=-np.inf, high=np.inf, shape=(69,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(67,), dtype=np.float64
            )
        self.frame_skip = 60
        MujocoEnv.__init__(
            self,
            os.getcwd()+"/scene.xml",
            self.frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # self.ref_trajectory = CassieTrajectory("reference_trajectories/cassie_walk/cassie_walking.mat")
        # print(self.ref_trajectory.time.shape)
        # exit()
        # print(self.ref_trajectory.time.shape)
        # exit()
        self.ref_qpos = np.load(
            'reference_trajectories/cassie_walk/old_cassie_reference_qpos_list.npy')
        self.ref_qvel = np.load(
            'reference_trajectories/cassie_walk/old_cassie_reference_qvel_list.npy')
        # Index from README.
        self.p_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.v_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        # print(self.ref_qpos.shape)

        self.timestamp = 0

        # initial_qpos, initial_qvel = self.ref_trajectory.state(0)
        self.init_qpos = self.ref_qpos[0]
        self.init_qvel = self.ref_qvel[0]
        self.reset_model()

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * \
            np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (
            not self.is_healthy) if self._terminate_when_unhealthy else False
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
                # com_inertia,
                # com_velocity,
                # actuator_forces,
                # external_contact_forces,
            )
        ).ravel()

    def PD(self, p_desired):
        # PD gain from old cassie env.
        kp = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
        kd = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
        p = self.data.qpos[self.p_index]
        v = self.data.qvel[self.v_index]
        v_desired = np.zeros(10)
        return kp * (p_desired - p) + kd * (v_desired - v)
    
    def _step_mujoco_simulation(self, ctrl, n_frames):
        # Set the control target, this userdata is used by PD control callback.
        self.data.userdata[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)
        
    # The action is now the target position.
    def step(self, action):
        self.timestamp += 1
        ref_qpos, ref_qvel = self.ref_qpos[self.timestamp], self.ref_qvel[self.timestamp]
        xy_position_before = mass_center(self.model, self.data)

        q_pos_modified = action + ref_qpos[self.p_index]
        # Simulate at 2000 Hz for frame_skip times.
        torque = self.PD(action)
        self._step_mujoco_simulation(q_pos_modified, self.frame_skip)

        xy_position_after = mass_center(self.model, self.data)
        # Transition happens here so time + 1

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = 0.1 * self.control_cost(torque)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        joint_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]

        joint_idx = [15, 16, 20, 29, 30, 34]
        joint_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]

        pos_index = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 21, 22, 23, 28, 29, 30, 34])
        vel_index = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 31])

        ref_pelvis_pos = ref_qpos[0:3]
        ref_pelvis_ori = ref_qpos[3:7]
        ref_joint_pos = ref_qpos[joint_idx]

        current_pelvis_pos = self.data.qpos[0:3]
        current_pelvis_ori = self.data.qpos[3:7]
        current_joint_pos = self.data.qpos[joint_idx]

        # the following imitation reward design is from Zhaoming's 2023 paper https://zhaomingxie.github.io/projects/Opt-Mimic/opt-mimic.pdf
        # sigmas = [0.05, 0.05, 0.3, 0.35, 0.3]
        sigmas = [1, 1, 1, 1, 1]
        reward_weights = [0.35, 0.35, 0.2, 0.1, 0.1]

        # reward for pelvis position difference
        r_0 = np.exp(- np.linalg.norm(ref_pelvis_pos -
                     current_pelvis_pos, ord=2))
        # reward for pelvis orientation difference
        r_1 = np.exp(- np.linalg.norm(ref_pelvis_ori -
                     current_pelvis_ori, ord=2))
        # reward for joint position difference
        r_2 = np.exp(- np.linalg.norm(ref_joint_pos -
                     current_joint_pos, ord=2))
        # # reward for action difference
        # r_3 = np.exp(-(np.linalg.norm(ref_torque - action, ord=2) ) / (2 * sigmas[3] ) )
        # # reward for maximum action difference
        # current_max_action = np.max(np.abs(action))
        # ref_max_action = np.max(np.abs(ref_torque))
        # r_4 = np.exp(-(np.abs(ref_max_action - current_max_action) **2) / (2 * sigmas[4] **2) )

        # + np.exp(-(np.linalg.norm(ref_qpos[:-1] - self.data.qpos))) * 1e1
        r_5 = np.exp(-(np.linalg.norm(ref_qvel[vel_index] -
                     self.data.qvel[vel_index], ord=2)) / (2 * 1)) * 1e1

        total_qpos_reward = np.exp(-np.linalg.norm(
            self.data.qpos[pos_index] - ref_qpos[pos_index], ord=2))

        total_reward = 0.1 * r_0
        total_reward += 0.1 * r_1
        total_reward += 0.2 * r_2
        total_reward += 0.2 * r_5
        total_reward += 0.3 * total_qpos_reward
        total_reward += 0.1 * forward_reward
        # total_reward -= 0.1 * ctrl_cost
        # + reward_weights[3] * r_3 + reward_weights[4] * r_4
        # total_reward = -np.linalg.norm(self.data.qpos - ref_qpos[:-1])-np.linalg.norm(action-ref_torque)
        # total_reward = np.exp(-np.linalg.norm(self.data.qpos - ref_qpos)) + np.exp(-np.linalg.norm(action-ref_torque))

        observation = self._get_obs()
        reward = total_reward

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

        observation = self._get_obs()
        self.timestamp = 0
        
        self.data.userdata = np.zeros(10) # Use userdata as target position.
        # Define a callback that modify the ctrl before mj_step.
        def PD_control_CB(model, data):
            kp = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
            kd = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
            p_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
            v_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
            p = data.qpos[p_index]
            v = data.qvel[v_index]
            p_desired = data.userdata[:]
            v_desired = np.zeros(10)
            data.ctrl[:] = kp * (p_desired - p) + kd * (v_desired - v)
            
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
