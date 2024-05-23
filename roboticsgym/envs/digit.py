import os
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import mujoco
from mujoco._functions import mj_rnePostConstraint
from mujoco._functions import mj_step

from jax import numpy as jnp

from roboticsgym.envs.reference_trajectories.loadDigit import DigitTrajectory

from roboticsgym.utils.transform3d import (
    euler2quat,
    inverse_quaternion,
    quaternion_product,
    quaternion2euler,
    rotate_by_quaternion,
    quat2yaw,
    euler2mat,
)

frameskip_global = 10

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
        "render_fps": round(2000 / frameskip_global),
    }

    def __init__(
        self,
        forward_reward_weight=0.1,
        ctrl_cost_weight=0.1,
        healthy_reward=0.1,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.5, 1.2),
        reset_noise_scale=1e-3,
        exclude_current_positions_from_observation=False,
        **kwargs,
    ):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self.start_time_stamp = 14000
        # self.start_time_stamp = 0
        self.timestamp = self.start_time_stamp
        self.frame_skip = frameskip_global
        self.render_fps = round(2000 / self.frame_skip)
        self.previous_action = np.zeros(20)

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(106,),
            dtype=np.float64,
        )

        MujocoEnv.__init__(
            self,
            os.getcwd() + "/roboticsgym/envs/xml/digit_scene.xml",
            self.frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        # overriding action space
        self.action_space = Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)  # type: ignore
        # print(self.action_space.shape)
        self.ref_trajectory = DigitTrajectory(
            os.getcwd()
            + "/roboticsgym/envs/"
            + "reference_trajectories/digit_state_20240514.csv"
        )
        # self.ref_trajectory_npz = np.load(
        #     os.getcwd()
        #     + "/roboticsgym/envs/reference_trajectories/digit_mujoco_controller_walking.npz"
        # )

        # self.ref_trajectory.qpos = self.ref_trajectory_npz["arr_0"]
        # self.ref_trajectory.qvel = self.ref_trajectory_npz["arr_1"]

        self.init_qpos, self.init_qvel = self.ref_trajectory.state(self.timestamp)

        # self.init_qvel = np.zeros_like(self.init_qvel)
        self.ref_qpos, self.ref_qvel = self.ref_trajectory.state(self.timestamp)
        self.next_ref_qpos, self.next_ref_qvel = self.ref_trajectory.state(
            self.timestamp + int(self.frame_skip / 2)
        )

        # will be used in PD
        self.target_velocity = np.zeros(20)

        # Index from README. The toes are actuated by motor A and B.
        self.p_index = [
            7,
            8,
            9,
            14,
            18,
            23,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            41,
            45,
            50,
            57,
            58,
            59,
            60,
        ]
        self.v_index = [
            6,
            7,
            8,
            12,
            16,
            20,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            36,
            40,
            44,
            50,
            51,
            52,
            53,
        ]
        self.kp = (
            np.array(
                [
                    1400,
                    1000,
                    1167,
                    1300,
                    533,
                    533,
                    500,
                    500,
                    500,
                    500,
                    1400,
                    1000,
                    1167,
                    1300,
                    533,
                    533,
                    500,
                    500,
                    500,
                    500,
                ]
            )
            # * 0.3
        )

        # self.kp = np.array(
        #     [
        #         100,
        #         100,
        #         88,
        #         96,
        #         50,
        #         50,
        #         50,
        #         50,
        #         50,
        #         50,
        #         100,
        #         100,
        #         88,
        #         96,
        #         50,
        #         50,
        #         50,
        #         50,
        #         50,
        #         50,
        #     ]
        # )

        # self.kd = np.array(
        #     [
        #         10.0,
        #         10.0,
        #         8.0,
        #         9.6,
        #         5.0,
        #         5.0,
        #         5.0,
        #         5.0,
        #         5.0,
        #         5.0,
        #         10.0,
        #         10.0,
        #         8.0,
        #         9.6,
        #         5.0,
        #         5.0,
        #         5.0,
        #         5.0,
        #         5.0,
        #         5.0,
        #     ]
        # )

        self.kd = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])

        self.recorded_qpos_index = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                14,
                15,
                16,
                17,
                18,
                23,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                41,
                42,
                43,
                44,
                45,
                50,
                55,
                56,
                57,
                58,
                59,
                60,
            ]
        )

        self.upper_body_index = np.array([30, 31, 32, 33, 57, 58, 59, 60])
        self.gear_ratio = self.model.actuator_gear[:, 0]

        self.reset_model()
        self.camera_name = "side"

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
        # is_healthy = is_healthy and (not np.allclose(self.data.cfrc_ext.flat.copy(), 0))
        return is_healthy

    @property
    def terminated(self):

        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        # print(qpos.shape, qvel.shape)

        joint_position = qpos[self.p_index]
        joint_velocity = qvel[self.v_index]

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        # print(com_inertia.shape, com_velocity.shape)

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        # print(actuator_forces.shape, external_contact_forces.shape)

        self.root_quat = qpos[3:7]
        roll, pitch, yaw = quaternion2euler(self.root_quat)
        base_rot = euler2mat(0, 0, yaw, "sxyz")
        self.root_lin_vel = np.transpose(base_rot).dot(qvel[0:3])
        self.root_ang_vel = np.transpose(base_rot).dot(qvel[3:6])
        # eular_angles = np.array([roll, pitch, yaw])
        return np.concatenate(
            (
                self.root_lin_vel,
                self.root_ang_vel,
                qpos[:7],
                joint_position,
                joint_velocity,
                self.next_ref_qpos[:7],
                self.next_ref_qvel[:6],
                self.next_ref_qpos[self.p_index],
                self.next_ref_qvel[self.v_index],
            )
        )

    # Basically PD control
    def _step_mujoco_simulation(self, target_position: np.array, n_frames: int) -> None:
        for _ in range(n_frames):
            motor_positions = self.data.actuator_length
            # current_position = motor_positions
            current_position = np.divide(motor_positions, self.gear_ratio)
            motor_velocities = self.data.actuator_velocity
            # current_velocity = motor_velocities
            current_velocity = np.divide(motor_velocities, self.gear_ratio)
            # Compute torque using PD gain
            torque = self.kp * (target_position - current_position) + self.kd * (
                self.target_velocity - current_velocity
            )
            torque = np.divide(torque, self.gear_ratio)
            self.data.ctrl[:] = torque.copy()

            mj_step(self.model, self.data)

            # As of MuJoCo 2.0, force-related quantities like cacc are not computed
            # unless there's a force sensor in the model.
            # See https://github.com/openai/gym/issues/1541
            mj_rnePostConstraint(self.model, self.data)

    def compute_torque(self, target_position, target_velocity):
        motor_positions = self.data.actuator_length
        current_position = motor_positions
        current_position = np.divide(motor_positions, self.gear_ratio)
        motor_velocities = self.data.actuator_velocity
        current_velocity = motor_velocities
        current_velocity = np.divide(motor_velocities, self.gear_ratio)

        # Compute torque using PD gain
        torque = self.kp * (target_position - current_position) + self.kd * (
            target_velocity - current_velocity
        )
        torque = np.divide(torque, self.gear_ratio)
        return torque

    def step(self, action):

        # Compute xy position before and after
        xy_position_before = mass_center(self.model, self.data)

        # Because the reference trajectory is at 1000Hz, while the simulation is 2000Hz,
        # we need to skip 30/2 frames
        self.timestamp += int(self.frame_skip / 2)
        # now we change to another traj which is 200hz
        # self.timestamp += 1

        # Create PD target
        # Get reference qpos and qvel
        self.ref_qpos, self.ref_qvel = self.ref_trajectory.state(self.timestamp)
        self.ref_torque = self.ref_trajectory.action(self.timestamp)
        self.next_ref_qpos, self.next_ref_qvel = self.ref_trajectory.state(
            self.timestamp + int(self.frame_skip / 2)
        )

        pd_target = action + self.ref_qpos[self.p_index]

        self._step_mujoco_simulation(pd_target, self.frame_skip)

        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # Compute reward using current qpos and qvel after the simulation
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        # print(
        #     "qpos",
        #     qpos[self.p_index],
        #     "ref qpos",
        #     self.ref_qpos[self.p_index],
        # )

        ctrl_cost = 0.1 * np.linalg.norm(action, ord=2)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        root_pos_tracking_rwd = np.exp(
            -10
            * np.linalg.norm(
                self.ref_qpos[:3] - qpos[:3],
            )
        )
        root_rot_tracking_rwd = np.exp(
            -10
            * np.linalg.norm(
                self.ref_qpos[3:7] - qpos[3:7],
            )
        )
        root_lin_vel_tracking_rwd = np.exp(
            -1
            * np.linalg.norm(
                self.ref_qvel[:3] - qvel[:3],
            )
        )

        root_ang_vel_tracking_rwd = np.exp(
            -1
            * np.linalg.norm(
                self.ref_qvel[3:6] - qvel[3:6],
            )
        )

        overall_pos_tracking_rwd = np.exp(
            -10
            * np.linalg.norm(
                self.ref_qpos[self.recorded_qpos_index] - qpos[self.recorded_qpos_index]
            )
        )

        foot_pos_tracking_rwd = -10 * np.linalg.norm(
            self.ref_qpos[
                [
                    18,
                    23,
                    45,
                    50,
                ]
            ]
            - qpos[
                [
                    18,
                    23,
                    45,
                    50,
                ]
            ],
        )
        upper_body_tracking_rwd = np.exp(
            -10
            * np.linalg.norm(
                self.init_qpos[self.upper_body_index] - qpos[self.upper_body_index],
            )
        )

        action_smoothing_rwd = np.exp(
            -1 * np.linalg.norm(self.previous_action - action)
        )

        tracking_reward = (
            root_pos_tracking_rwd
            + root_rot_tracking_rwd
            + root_lin_vel_tracking_rwd
            + root_ang_vel_tracking_rwd
            + overall_pos_tracking_rwd
            + foot_pos_tracking_rwd
            + upper_body_tracking_rwd
            + action_smoothing_rwd
        )

        # print(
        #     f"reward: root_pos_tracking_rwd: {root_pos_tracking_rwd}, root_rot_tracking_rwd: {root_rot_tracking_rwd}, root_lin_vel_tracking_rwd: {root_lin_vel_tracking_rwd}, root_ang_vel_tracking_rwd: {root_ang_vel_tracking_rwd}, overall_pos_tracking_rwd: {overall_pos_tracking_rwd}, foot_pos_tracking_rwd: {foot_pos_tracking_rwd}, upper_body_tracking_rwd: {upper_body_tracking_rwd}, action_smoothing_rwd: {action_smoothing_rwd}, ctrl_cost: {ctrl_cost}"
        # )
        self.previous_action = action

        # reward = (
        #     0.1 * forward_reward + 0.1 * healthy_reward + tracking_reward - ctrl_cost
        # )
        # reward = tracking_reward - ctrl_cost
        # reward = forward_reward + healthy_reward - ctrl_cost
        reward = tracking_reward + healthy_reward - ctrl_cost

        observation = self._get_obs()
        terminated = self.terminated
        info = {
            "reward_tracking": tracking_reward,
            "reward_root_pos": root_pos_tracking_rwd,
            "reward_root_rot": root_rot_tracking_rwd,
            "reward_root_lin_vel": root_lin_vel_tracking_rwd,
            "reward_root_ang_vel": root_ang_vel_tracking_rwd,
            "reward_overall_pos": overall_pos_tracking_rwd,
            "reward_foot_pos": foot_pos_tracking_rwd,
            "reward_upper_body": upper_body_tracking_rwd,
            "reward_action_smoothing": action_smoothing_rwd,
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
        # noise_low = -self._reset_noise_scale
        # noise_high = self._reset_noise_scale

        qpos = self.init_qpos
        qvel = self.init_qvel
        # self.set_state(qpos, qvel)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

        self.timestamp = self.start_time_stamp
        self.ref_qpos, self.ref_qvel = self.ref_trajectory.state(self.timestamp)
        self.next_ref_qpos, self.next_ref_qvel = self.ref_trajectory.state(
            self.timestamp + int(self.frame_skip / 2)
        )
        # self.next_ref_qpos, self.next_ref_qvel = self.ref_trajectory.state(
        #     self.timestamp + 1
        # )
        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
