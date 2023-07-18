import os
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import mujoco

# from cassie_m.cassiemujoco import CassieSim, CassieVis

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
        "render_fps": 36,
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
                low=-np.inf, high=np.inf, shape=(669,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(671,), dtype=np.float64
            )
        self.frame_skip = 60
        MujocoEnv.__init__(
            self,
            os.getcwd()+"/cassie.xml",
            self.frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.ref_trajectory = CassieTrajectory("reference_trajectories/cassie_walk/cassie_walking_from_stand.mat")

        self.timestamp = 0
        
        self._update_init_qpos()
        
        # print(f'final x pos {self.data.qpos[0]}, {self.ref_trajectory.qpos[0, 0]}')

        # self.sim = CassieSim(os.getcwd()+"/cassie.xml")
        # self.visual = True
        # if self.visual:
        #     self.vis = CassieVis(self.sim)
        # self.vis.draw(self.sim)

    def _update_init_qpos(self):
        # handcrafted init qpos
        qpos_init_cassiemujoco = np.array([0, 0, 1.01, 1, 0, 0, 0,
            0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
            -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
            -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
            -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968])
        self.init_qpos = qpos_init_cassiemujoco

        # self.init_qpos = self.ref_trajectory.qpos[0]
        # self.do_simulation(np.zeros(0), 1)
        self.init_qpos = np.array([0.04529737116916673, -0.15300356752917388, 0.9710095501646747, 1.0, 0.0, 0.0, 0.0, 0.04516039439535328, 0.0007669903939428207, 0.48967542963286953, 0.5366660119008494, -0.5459706642036749, 0.13716932320803393, 0.6285620114754674, -1.3017744461194523, -0.03886484136807136, 1.606101853366077, -0.7079960941663008, -1.786147490968169, 0.3175519006511133, -1.683487349162938, -0.04519107401111099, -0.0007669903939428207, 0.4898192403317338, 0.38803053590372555, -0.25971548696569596, 0.49875340077344466, -0.7302227155144948, -1.3018703199186952, -0.038780951793733864, 1.606065900691361, 0.49858954295641644, -1.6206843700546072, 0.12408356187240471, -1.6835283352121144])
        self.init_qvel = self.ref_trajectory.qvel[0]
        # self.set_state(self.init_qpos, self.init_qvel)
        self.reset()
        
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

    def step(self, action):
        # xy_position_before = mass_center(self.model, self.data)
        # self.do_simulation(action, self.frame_skip)
        # xy_position_after = mass_center(self.model, self.data)

        # xy_velocity = (xy_position_after - xy_position_before) / self.dt
        # x_velocity, y_velocity = xy_velocity

        # ctrl_cost = self.control_cost(action)

        # forward_reward = self._forward_reward_weight * x_velocity
        # healthy_reward = self.healthy_reward

        # rewards = forward_reward + healthy_reward

        # observation = self._get_obs()
        # reward = rewards - ctrl_cost
        # terminated = self.terminated
        # info = {
        #     "reward_linvel": forward_reward,
        #     "reward_quadctrl": -ctrl_cost,
        #     "reward_alive": healthy_reward,
        #     "x_position": xy_position_after[0],
        #     "y_position": xy_position_after[1],
        #     "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
        #     "x_velocity": x_velocity,
        #     "y_velocity": y_velocity,
        #     "forward_reward": forward_reward,
        # }

        # if self.render_mode == "human":
        #     self.render()
        # return observation, reward, terminated, False, info
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)
        # Transition happens here so time + 1
        self.timestamp += 1

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = 0.1 * self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        joint_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        
        # if frameskip = 5, we don't need to multiply 6
        ref_qpos, ref_qvel = self.ref_trajectory.state(self.timestamp * self.frame_skip)
        ref_pelvis_pos = ref_qpos[0:3]
        ref_pelvis_ori = ref_qpos[3:7]
        ref_joint_pos = ref_qpos[joint_idx]

        current_pelvis_pos = self.data.qpos[0:3]
        current_pelvis_ori = self.data.qpos[3:7]
        current_joint_pos = self.data.qpos[joint_idx]
        
        ref_mpos, ref_mvel, ref_torque = self.ref_trajectory.action(self.timestamp * self.frame_skip)

        # the following imitation reward design is from Zhaoming's 2023 paper https://zhaomingxie.github.io/projects/Opt-Mimic/opt-mimic.pdf
        # sigmas = [0.05, 0.05, 0.3, 0.35, 0.3]
        sigmas = [1, 1, 1, 1, 1]
        reward_weights = [0.3, 0.3, 0.2, 0.1, 0.1] 

        # reward for pelvis position difference
        r_0 = np.exp(- (np.linalg.norm(ref_pelvis_pos - current_pelvis_pos, ord=2) **2 )/ (2 * sigmas[0] **2 ) ) 
        # reward for pelvis orientation difference
        r_1 = np.exp(- (np.linalg.norm(ref_pelvis_ori - current_pelvis_ori, ord=2) **2) / (2 * sigmas[1] **2 ) ) 
        # reward for joint position difference
        r_2 = np.exp(-(np.linalg.norm(ref_joint_pos - current_joint_pos, ord=2) **2) / (2 * sigmas[2] **2 ) ) 
        # reward for action difference
        r_3 = np.exp(-(np.linalg.norm(ref_torque - action, ord=2) ) / (2 * sigmas[3] ) ) 
        # reward for maximum action difference
        current_max_action = np.max(np.abs(action))
        ref_max_action = np.max(np.abs(ref_torque))
        r_4 = np.exp(-(np.abs(ref_max_action - current_max_action) **2) / (2 * sigmas[4] **2) ) 

        r_5 = np.exp(-(np.linalg.norm(ref_qvel - self.data.qvel, ord=2) ) / (2 * 1 ) ) * 1e1 # + np.exp(-(np.linalg.norm(ref_qpos[:-1] - self.data.qpos))) * 1e1

        total_reward = reward_weights[0] * r_0 + reward_weights[1] * r_1 + reward_weights[2] * r_2 + reward_weights[3] * r_3 + reward_weights[4] * r_4 #+ 0.3 * r_5
        # total_reward = -np.linalg.norm(self.data.qpos - ref_qpos[:-1])-np.linalg.norm(action-ref_torque)
        total_reward = np.exp(-np.linalg.norm(self.data.qpos - ref_qpos)) + np.exp(-np.linalg.norm(action-ref_torque))
        
        observation = self._get_obs()
        reward = total_reward 
        # reward = total_reward + forward_reward + healthy_reward - ctrl_cost
        # reward = forward_reward + healthy_reward - ctrl_cost
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

        # print(ref_qpos[:3], current_pelvis_pos[:3])
        # print(reward)
        # if terminated:
            # exit()
            # print(f'final x pos {xy_position_after[0]:.2f}, {ref_pelvis_pos[0]:.2f}, {current_pelvis_pos[0]:.2f}')
        # print(f'{r_0:.2e}, {r_1:.2e}, {r_2:.2e}, {r_3:.2e}, {r_4:.2e}, {r_5:.2e}')
        # print(f'{self.data.qpos[:3]}, {ref_pelvis_pos}')
        # import time
        # time.sleep(0.01)

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
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
