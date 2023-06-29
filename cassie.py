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
        "render_fps": 67,
    }

    def __init__(
            self,
            forward_reward_weight=1.25,
            ctrl_cost_weight=0.1,
            healthy_reward=5.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.7, 2.0),
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
        self.frameskip = 30
        MujocoEnv.__init__(
            self,
            os.getcwd()+"/scene.xml",
            self.frameskip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        
        self.ref_trajectory = CassieTrajectory("reference_trajectories/cassie_walk/cassie_walking_from_stand.mat")

        self.timestamp = 65

        initial_qpos, initial_qvel = self.ref_trajectory.state(self.timestamp * self.frame_skip)
        self.init_qpos = initial_qpos
        self.init_qpos = np.array( [0.7168356984991782, 0.21534483766233287, 0.9323887592900508, 0.9999999264974779, -0.0002901375476954636, 0.00013202243630762556, -0.0002130617714944957, -0.03982478079625715, 0.0211525159580161, 0.7244190781589216, 0.017052475696354544, 0.9520544633328609, -0.30403324295240536, -0.02941596676036783, -1.358681181241618, -0.02049076550104647, 1.6329803035030004, -0.011073575029023954, -1.7530196679331227, 1.733781197099566, -1.856552584933119, 0.006832282644159905, -0.017280732532940272, 0.3852306316070201, -0.7984661397509015, -0.5377991945089965, 0.167180037668535, 0.21277848824990653, -1.2837415386581086, -0.07751563423636267, 1.6674075108118946, -0.0368339493559848, -1.5389410762823728, 1.5204427500858477, -1.5628833194785936]) #the 65 * 5 = 325th frame of the reference trajectory
        self.init_qvel = initial_qvel
        self.init_qvel = np.array([0.6747320619641718, 0.2090707285249807, -0.24060797483179508, -0.2880849285279525, 0.15982699851591325, -0.12528982482833495, 0.2019895410859575, -0.12011558661951015, -2.1371205058577534, -0.001973926630127514, -3.3500663038668734, 0.046384842673533475, 3.139542496927967, -2.1174284387437474, 2.0149343418219177, -2.3090546279018462, 1.4157711330477927, -1.2307522395721413, -2.1584881615936338, 0.10720791732915282, -0.22555645280024142, 0.3715655385517533, 0.004758047645511424, 0.6633032592511838, -0.23061693624232485, -2.407799781121452, 3.4704447285585016, -1.8825803940069632, -0.4085142325556717, 1.0250941030366612, -0.9983749965592585, 0.029746922964266793])
        # self.reset()
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
        # total_reward = np.exp(-np.linalg.norm(self.data.qpos - ref_qpos)) + np.exp(-np.linalg.norm(action-ref_torque))
        
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

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
