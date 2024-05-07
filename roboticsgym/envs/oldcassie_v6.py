from ast import Dict
import os
import random
import pickle
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import utils
import datetime
from typing import Any, Union
from gymnasium import spaces

from roboticsgym.envs.old_cassie.cassie_m.cassiemujoco import CassieSim, CassieVis
from roboticsgym.envs.old_cassie.cassie_env.loadstep import CassieTrajectory
from roboticsgym.envs.old_cassie.cassie_env.quaternion_function import (
    euler2quat,
    inverse_quaternion,
    quaternion_product,
    quaternion2euler,
    rotate_by_quaternion,
    quat2yaw,
)
from roboticsgym.envs.old_cassie.cassie_m.cassiemujoco import (
    pd_in_t,
    state_out_t,
    CassieSim,
    CassieVis,
)
from roboticsgym.envs.old_cassie.cassie_m.cassiemujoco_ctypes import (
    cassie_sim_init,
    cassie_sim_free,
)

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


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


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
        xml_path="roboticsgym/envs/old_cassie/cassie_m/model/0cassie.xml",
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.8, 2.0),
        reset_noise_scale=1e-3,
        random_state=False,
        visual=False,
        visual_record=False,
        record_for_reward_inference=False,
        random_terrain=False,
        enhanced_reward: str = "naive",
        difficulty_level: int = 1,
        terrain_file_path: str = "terrain_sine_t0.png",
        log_file_path: str | None = None,
        domain_randomization_scale: float = 0.0,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_path,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            random_state,
            visual,
            visual_record,
            record_for_reward_inference,
            random_terrain,
            enhanced_reward,
            difficulty_level,
            terrain_file_path,
            log_file_path,
            **kwargs,
        )
        self.model_xml_path = xml_path
        self.random_terrain = random_terrain
        self.sim = CassieSim(
            self.model_xml_path,
            terrain=self.random_terrain,
            difficulty_level=difficulty_level,
        )
        self.visual = visual
        self.visual_record = visual_record
        self.random_state = random_state
        self.enhance_reward = enhanced_reward
        self.log_file_path = log_file_path
        if self.render_mode is not None:
            print(self.render_mode)
        if self.visual or self.render_mode is not None:
            print("Visualizing")
            self.vis = CassieVis(self.sim)
            if self.visual_record:
                t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                if self.log_file_path is not None:
                    self.vis.init_recording(f"logs/{self.log_file_path}/{t}")
                else:
                    self.vis.init_recording(f"logs/videos/{t}")
                self.vis.set_cam("cassie-pelvis", 3, 90, -20)
        self.whole_state_buffer = []
        self.useful_recorded_data = []
        self.state_buffer = []
        self.domain_randomization_scale = domain_randomization_scale
        self.delay = False
        self.buffer_size = 20
        self.noisy = False
        self.cassie_state = state_out_t()
        self.u = pd_in_t()
        self.orientation = 0
        self.foot_forces = np.ones(2) * 500
        self.max_phase = 28
        self.control_rate = 60
        self.time_limit = 400 * 60 / self.control_rate
        self.trajectory = CassieTrajectory(
            "roboticsgym/envs/old_cassie/trajectory/stepdata.bin"
        )

        self.x_vel_sum = 0
        with open(
            "roboticsgym/envs/old_cassie/trajectory/stepping_trajectory_Nov", "rb"
        ) as fp:
            self.step_in_place_trajectory = pickle.load(fp)
        with open(
            "roboticsgym/envs/old_cassie/trajectory/backward_trajectory_Nov", "rb"
        ) as fp:
            self.backward_trajectory = pickle.load(fp)
        self.side_speed = 0
        for i in range(1682):
            self.step_in_place_trajectory[i][0][0] = 0
            self.step_in_place_trajectory[i][0][1] = 0
            self.step_in_place_trajectory[i][0][2] = 1.05
            self.step_in_place_trajectory[i][0][3] = 1
            self.step_in_place_trajectory[i][0][4] = 0
            self.step_in_place_trajectory[i][0][5] = 0
            self.step_in_place_trajectory[i][0][6] = 0
            self.step_in_place_trajectory[i][0][7] = 0
            self.step_in_place_trajectory[i][0][8] = 0
            self.step_in_place_trajectory[i][0][21] = 0
            self.step_in_place_trajectory[i][0][22] = 0
            self.step_in_place_trajectory[i][1][6] = 0
            self.step_in_place_trajectory[i][1][7] = 0
            self.step_in_place_trajectory[i][1][19] = 0
            self.step_in_place_trajectory[i][1][20] = 0
            self.backward_trajectory[i][1][3] = 0
            self.backward_trajectory[i][1][4] = 0
            self.backward_trajectory[i][1][5] = 0
            self.backward_trajectory[i][0][0] = self.trajectory.qpos[i][0] * -1
            self.backward_trajectory[i][0][1] = 0
            self.backward_trajectory[i][0][2] = 1.05
            self.backward_trajectory[i][0][3] = 1
            self.backward_trajectory[i][0][4] = 0
            self.backward_trajectory[i][0][5] = 0
            self.backward_trajectory[i][0][6] = 0
            self.backward_trajectory[i][0][7] = 0
            self.backward_trajectory[i][0][8] = 0
            self.backward_trajectory[i][0][21] = 0
            self.backward_trajectory[i][0][22] = 0
            self.backward_trajectory[i][1][6] = 0
            self.backward_trajectory[i][1][7] = 0
            self.backward_trajectory[i][1][19] = 0
            self.backward_trajectory[i][1][20] = 0
            self.backward_trajectory[i][1][3] = 0
            self.backward_trajectory[i][1][4] = 0
            self.backward_trajectory[i][1][5] = 0
            self.trajectory.qpos[i][2] = 1.05

        for i in range(841):
            self.backward_trajectory[i][0][7:21] = np.copy(
                self.backward_trajectory[i + 841][0][21:35]
            )
            self.backward_trajectory[i][0][12] = -self.backward_trajectory[i][0][12]
            self.backward_trajectory[i][0][21:35] = np.copy(
                self.backward_trajectory[i + 841][0][7:21]
            )
            self.backward_trajectory[i][0][26] = -self.backward_trajectory[i][0][26]

            self.step_in_place_trajectory[i][0][7:21] = np.copy(
                self.step_in_place_trajectory[i + 841][0][21:35]
            )
            self.step_in_place_trajectory[i][0][12] = -self.step_in_place_trajectory[i][
                0
            ][12]
            self.step_in_place_trajectory[i][0][21:35] = np.copy(
                self.step_in_place_trajectory[i + 841][0][7:21]
            )
            self.step_in_place_trajectory[i][0][26] = -self.step_in_place_trajectory[i][
                0
            ][26]

            self.trajectory.qpos[i][7:21] = np.copy(
                self.trajectory.qpos[(i + 841)][21:35]
            )
            self.trajectory.qpos[i][12] = -self.trajectory.qpos[i][12]
            self.trajectory.qpos[i][21:35] = np.copy(
                self.trajectory.qpos[(i + 841)][7:21]
            )
            self.trajectory.qpos[i][26] = -self.trajectory.qpos[i][26]

        self.time = 0
        self.phase = 0
        self.counter = 0
        self.rew_ref = 0
        self.rew_spring = 0
        self.rew_ori = 0
        self.rew_vel = 0
        self.rew_termin = 0
        self.rew_cur = 0
        self.reward = 0
        self.rew_ref_buf = 0
        self.rew_spring_buf = 0
        self.rew_ori_buf = 0
        self.rew_vel_buf = 0
        self.rew_termin_buf = 0
        self.rew_cur_buf = 0
        self.reward_buf = 0
        self.omega_buf = 0
        self.record_state = False
        self.recorded = False
        self.recorded_state = []
        self.max_phase = 28
        self.control_rate = 60
        self.time_limit = 600
        # self.speed = (random.randint(-10, 10)) / 10.0
        self.speed = 1

        self.P = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])

        self.record_for_reward_inference = record_for_reward_inference

        self.first_phase_pos_index = np.array(
            [2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 21, 22, 23, 28, 29, 30, 34]
        )
        self.first_phase_vel_index = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 31]
        )

        self.second_phase_pos_index = np.array(
            [2, 3, 4, 5, 6, 21, 22, 23, 28, 29, 30, 34, 7, 8, 9, 14, 15, 16, 20]
        )
        self.second_phase_vel_index = np.array(
            [0, 1, 2, 3, 4, 5, 19, 20, 21, 25, 26, 27, 31, 6, 7, 8, 12, 13, 14, 18]
        )
        # Set state, observation and action space
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._get_obs()["state"].shape[0],),
                    dtype=np.float64,
                ),
                "observation": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._get_obs()["observation"].shape[0],),
                    dtype=np.float64,
                ),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

        """
        Position [1], [2] 				-> Pelvis y, z
                 [3], [4], [5], [6] 	-> Pelvis Orientation qw, qx, qy, qz
                 [7], [8], [9]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
                 [14]					-> Left Knee   	(Motor[3])
                 [15]					-> Left Shin   	(Joint[0])
                 [16]					-> Left Tarsus 	(Joint[1])
                 [20]					-> Left Foot   	(Motor[4], Joint[2])
                 [21], [22], [23]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
                 [28]					-> Rigt Knee   	(Motor[8])
                 [29]					-> Rigt Shin   	(Joint[3])
                 [30]					-> Rigt Tarsus 	(Joint[4])
                 [34]					-> Rigt Foot   	(Motor[9], Joint[5])
        """

        """
        Velocity [0], [1], [2] 			-> Pelvis x, y, z
                 [3], [4], [5]		 	-> Pelvis Orientation wx, wy, wz
                 [6], [7], [8]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
                 [12]					-> Left Knee   	(Motor[3])
                 [13]					-> Left Shin   	(Joint[0])
                 [14]					-> Left Tarsus 	(Joint[1])
                 [18]					-> Left Foot   	(Motor[4], Joint[2])
                 [19], [20], [21]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
                 [25]					-> Rigt Knee   	(Motor[8])
                 [26]					-> Rigt Shin   	(Joint[3])
                 [27]					-> Rigt Tarsus 	(Joint[4])
                 [31]					-> Rigt Foot   	(Motor[9], Joint[5])
        """

    def get_kin_state(self):
        if self.speed < 0:
            interpolate = self.speed * -1
            phase = self.phase
            pose = np.copy(
                self.backward_trajectory[phase * self.control_rate][0]
            ) * interpolate + (1 - interpolate) * np.copy(
                self.step_in_place_trajectory[phase * self.control_rate][0]
            )
            pose[0] += (
                (
                    self.backward_trajectory[(self.max_phase - 1) * self.control_rate][
                        0
                    ][0]
                    - self.backward_trajectory[0][0][0]
                )
                * self.counter
                * (-self.speed)
            )
            pose[1] = 0
            vel = np.copy(self.backward_trajectory[phase * self.control_rate][1])
            vel[0] *= -self.speed
        elif self.speed <= 1.0:
            interpolate = self.speed * 1
            pose = np.copy(
                self.trajectory.qpos[self.phase * self.control_rate]
            ) * interpolate + (1 - interpolate) * np.copy(
                self.step_in_place_trajectory[self.phase * self.control_rate][0]
            )
            pose[0] = self.trajectory.qpos[self.phase * self.control_rate, 0]
            pose[0] *= self.speed
            pose[0] += (
                (self.trajectory.qpos[1681, 0] - self.trajectory.qpos[0, 0])
                * self.counter
                * self.speed
            )
            pose[1] = 0
            vel = np.copy(self.trajectory.qvel[self.phase * self.control_rate])
            vel[0] *= self.speed
        else:
            pose = np.copy(self.trajectory.qpos[self.phase * self.control_rate])
            pose[0] = self.trajectory.qpos[self.phase * self.control_rate, 0]
            pose[0] *= self.speed
            pose[0] += (
                (self.trajectory.qpos[1681, 0] - self.trajectory.qpos[0, 0])
                * self.counter
                * self.speed
            )
            pose[1] = 0
            vel = np.copy(self.trajectory.qvel[self.phase * self.control_rate])
            vel[0] *= self.speed
        # print("vel", vel[0])
        pose[1] = (
            self.side_speed
            * (self.counter * self.max_phase + self.phase)
            * self.control_rate
            * 0.0005
        )
        pose[3] = 1
        pose[4:7] = 0
        pose[7] = 0
        pose[8] = 0
        pose[21] = 0
        pose[22] = 0
        vel[1] = self.side_speed
        vel[6] = 0
        vel[7] = 0
        vel[19] = 0
        vel[20] = 0
        return pose, vel

    def get_kin_next_state(self):
        if self.speed < 0:
            phase = self.phase + 1
            counter = self.counter

            if phase == self.max_phase:
                phase = 0
                counter = self.counter + 1
            phase = self.max_phase - phase
            pose = np.copy(self.trajectory.qpos[phase * self.control_rate])
            vel = np.copy(self.trajectory.qvel[phase * self.control_rate])
            pose[0] += (
                (self.trajectory.qpos[1681, 0] - self.trajectory.qpos[0, 0])
                * counter
                * self.speed
            )
            # print(pose[0])
            pose[1] = 0
            vel[0] *= self.speed
            pose[1] = (
                self.side_speed
                * (counter * self.max_phase + self.phase)
                * self.control_rate
                * 0.0005
            )

        else:
            phase = self.phase + 1
            counter = self.counter

            if phase == self.max_phase:
                phase = 0
                counter = self.counter + 1
            pose = np.copy(self.trajectory.qpos[phase * self.control_rate])
            pose[0] *= self.speed
            vel = np.copy(self.trajectory.qvel[phase * self.control_rate])
            pose[0] += (
                (self.trajectory.qpos[1681, 0] - self.trajectory.qpos[0, 0])
                * counter
                * self.speed
            )
            pose[1] = 0
            vel[0] *= self.speed
            pose[1] = (
                self.side_speed
                * (counter * self.max_phase + self.phase)
                * self.control_rate
                * 0.0005
            )
        pose[3] = 1
        pose[4:7] = 0
        pose[7] = 0
        pose[8] = 0
        pose[21] = 0
        pose[22] = 0
        vel[1] = self.side_speed
        vel[6] = 0
        vel[7] = 0
        vel[19] = 0
        vel[20] = 0
        return pose, vel

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.time != 0:
            self.rew_ref_buf = self.rew_ref / self.time
            self.rew_spring_buf = self.rew_spring / self.time
            self.rew_ori_buf = self.rew_ori / self.time
            self.rew_vel_buf = self.rew_vel / self.time
            self.reward_buf = self.reward  # / self.time
            self.time_buf = self.time

        self.rew_ref = 0
        self.rew_spring = 0
        self.rew_ori = 0
        self.rew_vel = 0
        self.rew_termin = 0
        self.rew_cur = 0
        self.reward = 0
        self.omega = 0
        self.height_rec = []

        self.orientation = 0
        # self.speed = (random.randint(-10, 10)) / 10.0
        self.speed = 1
        orientation = self.orientation + random.randint(-20, 20) * np.pi / 100
        orientation = 0
        quaternion = euler2quat(z=orientation, y=0, x=0)
        # self.phase = random.randint(0, 27)
        self.phase = 0
        self.time = 0
        self.counter = 0
        cassie_sim_free(self.sim.c)
        self.sim.c = cassie_sim_init(self.model_xml_path.encode("utf-8"), False)

        qpos, qvel = self.get_kin_state()
        qpos[3:7] = quaternion
        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)
        self.cassie_state = self.sim.step_pd(self.u)

        return self._get_obs(), dict()

    def set_domain_randomization_scale(self, domain_randomization_scale: float):
        # print("setting new domain randomization scale")
        self.domain_randomization_scale = domain_randomization_scale

    def apply_randomization(self, obs: np.ndarray) -> np.ndarray:
        if self.domain_randomization_scale == 0:
            return obs
        else:
            # return obs.copy() + np.random.normal(
            #     scale=self.domain_randomization_scale * np.abs(obs), size=obs.shape
            # )
            return obs.copy() + +np.random.uniform(
                low=-self.domain_randomization_scale * np.abs(obs),
                high=self.domain_randomization_scale * np.abs(obs),
                size=obs.shape,
            )

    def _get_obs(self):
        state = self.cassie_state
        ref_pos, ref_vel = self.get_kin_next_state()

        if self.phase < 14:
            quaternion = euler2quat(z=self.orientation, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)
            new_orientation = quaternion_product(
                iquaternion, state.pelvis.orientation[:]
            )
            # print(new_orientation)
            new_translationalVelocity = rotate_by_quaternion(
                state.pelvis.translationalVelocity[:], iquaternion
            )
            # print(new_translationalVelocity)
            new_translationalAcceleration = rotate_by_quaternion(
                state.pelvis.translationalAcceleration[:], iquaternion
            )
            # new_rotationalVelocity = rotate_by_quaternion(
            #     state.pelvis.rotationalVelocity[:], quaternion
            # )

            useful_state = np.copy(
                np.concatenate(
                    [
                        [state.pelvis.position[2] - state.terrain.height],
                        new_orientation[:],
                        state.motor.position[:],
                        new_translationalVelocity[:],
                        state.pelvis.rotationalVelocity[:],
                        state.motor.velocity[:],
                        new_translationalAcceleration[:],
                        state.joint.position[:],
                        state.joint.velocity[:],
                    ]
                )
            )

            # make useful_obs to be only include joint position and velocity
            obs = np.concatenate(
                [
                    state.joint.position[:],
                    state.joint.velocity[:],
                    new_orientation[:],
                    state.motor.position[:],
                    state.motor.velocity[:],
                ]
            )
            obs = self.apply_randomization(obs)
            obs = np.concatenate(
                [
                    obs,
                    ref_pos[self.first_phase_pos_index],
                    ref_vel[self.first_phase_vel_index],
                ]
            )

            state = np.concatenate(
                [
                    useful_state,
                    ref_pos[self.first_phase_pos_index],
                    ref_vel[self.first_phase_vel_index],
                ]
            )
            return {"state": state, "observation": obs}
        else:
            ref_vel[1] = -ref_vel[1]
            euler = quaternion2euler(ref_pos[3:7])
            euler[0] = -euler[0]
            euler[2] = -euler[2]
            ref_pos[3:7] = euler2quat(z=euler[2], y=euler[1], x=euler[0])
            quaternion = euler2quat(z=-self.orientation, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)

            pelvis_euler = quaternion2euler(np.copy(state.pelvis.orientation[:]))
            pelvis_euler[0] = -pelvis_euler[0]
            pelvis_euler[2] = -pelvis_euler[2]
            pelvis_orientation = euler2quat(
                z=pelvis_euler[2], y=pelvis_euler[1], x=pelvis_euler[0]
            )

            translational_velocity = np.copy(state.pelvis.translationalVelocity[:])
            translational_velocity[1] = -translational_velocity[1]

            translational_acceleration = np.copy(
                state.pelvis.translationalAcceleration[:]
            )
            translational_acceleration[1] = -translational_acceleration[1]

            rotational_velocity = np.copy(state.pelvis.rotationalVelocity)
            rotational_velocity[0] = -rotational_velocity[0]
            rotational_velocity[2] = -rotational_velocity[2]

            motor_position = np.zeros(10)
            motor_position[0:5] = np.copy(state.motor.position[5:10])
            motor_position[5:10] = np.copy(state.motor.position[0:5])
            motor_position[0] = -motor_position[0]
            motor_position[1] = -motor_position[1]
            motor_position[5] = -motor_position[5]
            motor_position[6] = -motor_position[6]

            motor_velocity = np.zeros(10)
            motor_velocity[0:5] = np.copy(state.motor.velocity[5:10])
            motor_velocity[5:10] = np.copy(state.motor.velocity[0:5])
            motor_velocity[0] = -motor_velocity[0]
            motor_velocity[1] = -motor_velocity[1]
            motor_velocity[5] = -motor_velocity[5]
            motor_velocity[6] = -motor_velocity[6]

            joint_position = np.zeros(6)
            joint_position[0:3] = np.copy(state.joint.position[3:6])
            joint_position[3:6] = np.copy(state.joint.position[0:3])

            joint_velocity = np.zeros(6)
            joint_velocity[0:3] = np.copy(state.joint.velocity[3:6])
            joint_velocity[3:6] = np.copy(state.joint.velocity[0:3])

            left_toeForce = np.copy(state.rightFoot.toeForce[:])
            left_toeForce[1] = -left_toeForce[1]
            left_heelForce = np.copy(state.rightFoot.heelForce[:])
            left_heelForce[1] = -left_heelForce[1]

            right_toeForce = np.copy(state.leftFoot.toeForce[:])
            right_toeForce[1] = -right_toeForce[1]
            right_heelForce = np.copy(state.leftFoot.heelForce[:])
            right_heelForce[1] = -right_heelForce[1]

            new_orientation = quaternion_product(iquaternion, pelvis_orientation)
            new_translationalVelocity = rotate_by_quaternion(
                translational_velocity, iquaternion
            )
            new_translationalAcceleration = rotate_by_quaternion(
                translational_acceleration, iquaternion
            )
            # new_rotationalVelocity = rotate_by_quaternion(
            #     rotational_velocity, quaternion
            # )

            useful_state = np.copy(
                np.concatenate(
                    [
                        [state.pelvis.position[2] - state.terrain.height],
                        new_orientation[:],
                        motor_position,
                        new_translationalVelocity[:],
                        rotational_velocity,
                        motor_velocity,
                        new_translationalAcceleration[:],
                        joint_position,
                        joint_velocity,
                    ]
                )
            )

            # make useful_obs to be only include joint position and velocity

            obs = np.concatenate(
                [
                    joint_position,
                    joint_velocity,
                    new_orientation[:],
                    motor_position,
                    motor_velocity,
                ]
            )
            obs = self.apply_randomization(obs)
            obs = np.concatenate(
                [
                    obs,
                    ref_pos[self.second_phase_pos_index],
                    ref_vel[self.second_phase_vel_index],
                ]
            )

            state = np.concatenate(
                [
                    useful_state,
                    ref_pos[self.second_phase_pos_index],
                    ref_vel[self.second_phase_vel_index],
                ]
            )
            return {"state": state, "observation": obs}

    def step_simulation(self, action):
        pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        ref_pos, _ = self.get_kin_next_state()
        self.current_action = np.array(action)
        if self.phase < 14:
            target = action + ref_pos[pos_index]
        else:
            mirror_action = np.zeros(10)
            mirror_action[0:5] = np.copy(action[5:10])
            mirror_action[5:10] = np.copy(action[0:5])
            mirror_action[0] = -mirror_action[0]
            mirror_action[1] = -mirror_action[1]
            mirror_action[5] = -mirror_action[5]
            mirror_action[6] = -mirror_action[6]
            target = mirror_action + ref_pos[pos_index]

        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.torque[i] = 0  # Feedforward torque
            self.u.leftLeg.motorPd.pTarget[i] = target[i]
            self.u.leftLeg.motorPd.pGain[i] = self.P[i]
            self.u.leftLeg.motorPd.dTarget[i] = 0
            self.u.leftLeg.motorPd.dGain[i] = self.D[i]
            self.u.rightLeg.motorPd.torque[i] = 0  # Feedforward torque
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i + 5]
            self.u.rightLeg.motorPd.dTarget[i] = 0
            self.u.rightLeg.motorPd.dGain[i] = self.D[i + 5]

        self.state_buffer.append(self.sim.step_pd(self.u))
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)

        self.cassie_state = self.state_buffer[len(self.state_buffer) - 1]

    def compute_reward(
        self,
    ):
        ref_pos, ref_vel = self.get_kin_state()
        cur_qpos = np.array(self.sim.qpos())
        cur_qvel = np.array(self.sim.qvel())
        weight = np.array([0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05])
        joint_penalty = 0
        joint_index = np.array([7, 8, 9, 14, 20, 21, 22, 23, 28, 34])
        vel_index = np.array([6, 7, 8, 12, 18, 19, 20, 21, 25, 31])

        errors = weight * (ref_pos[joint_index] - cur_qpos[joint_index]) ** 2
        joint_penalty = np.sum(errors * 30)

        pelvis_pos = np.copy(self.cassie_state.pelvis.position[:])
        com_penalty = (
            (pelvis_pos[0] - ref_pos[0]) ** 2
            + (pelvis_pos[1] - ref_pos[1]) ** 2
            + (cur_qvel[2]) ** 2
        )

        # yaw = quat2yaw(self.sim.qpos()[3:7])

        # orientation_penalty = (self.sim.qpos()[4])**2+(self.sim.qpos()[5])**2+(yaw - self.orientation)**2
        orientation_penalty = np.power(np.linalg.norm(cur_qpos[3:7] - ref_pos[3:7]), 2)

        overall_pos_penalty = np.power(np.linalg.norm(cur_qpos - ref_pos), 2)

        spring_penalty = (cur_qpos[15]) ** 2 + (cur_qpos[29]) ** 2
        spring_penalty *= 1000

        # speed_penalty = (self.sim.qvel()[0] - ref_vel[0])**2 + (self.sim.qvel()[1] - ref_vel[1])**2
        speed_penalty = np.power(
            np.linalg.norm(cur_qvel[vel_index] - ref_vel[vel_index]), 2
        )
        total_reward = (
            0.3 * np.exp(-joint_penalty)
            + 0.3 * np.exp(-com_penalty)
            + 0.2 * np.exp(-10 * orientation_penalty)
            + 0.1 * np.exp(-overall_pos_penalty)
            + 0.1 * np.exp(-speed_penalty)
        )

        forward_reward = 0.25 * cur_qvel[0]

        control_cost = 0.25 * np.power(np.linalg.norm(self.current_action), 2)

        total_reward += forward_reward - control_cost

        self.rew_ref += 0.5 * np.exp(-joint_penalty)
        self.rew_spring += 0.1 * np.exp(-spring_penalty)
        self.rew_ori += 0.1 * np.exp(-orientation_penalty)
        self.rew_vel += 0.3 * np.exp(-com_penalty)
        self.reward += total_reward

        return total_reward

    def step(self, action):
        self.current_action = action
        for _ in range(self.control_rate):
            self.step_simulation(action)

        center_of_mass_vel = self.sim.center_of_mass_velocity()  # list of 3
        if self.record_for_reward_inference:
            self.whole_state_buffer.append(self._get_obs())
            left_foot_force, right_foot_force = self.sim.get_foot_forces()
            center_of_mass_pos = self.sim.center_of_mass_position()

            center_of_mass_angular_momentum = self.sim.angular_momentum()  # list of 3
            # 3 by 3 matrix, list of 9
            center_of_mass_centroid_inertia = self.sim.centroid_inertia()
            feet_pos = self.sim.foot_pos()
            self.useful_recorded_data.append(
                [left_foot_force, right_foot_force]
                + center_of_mass_pos
                + feet_pos
                + center_of_mass_vel
                + center_of_mass_angular_momentum
                + center_of_mass_centroid_inertia
            )
        self.x_vel_sum += center_of_mass_vel[0]

        height = self.sim.qpos()[2]
        self.time += 1
        self.phase += 1

        if self.phase >= self.max_phase:
            self.phase = 0
            self.counter += 1
        # print("height", height)

        done = not (height > 0.4 and height < 100.0) or self.time >= self.time_limit
        # yaw = quat2yaw(self.sim.qpos()[3:7])
        if self.visual:
            self.render()

        reward = self.compute_reward()
        if reward < 0.3:
            done = True
        # if done and self.record_for_reward_inference:
        #     # save the whole state_buffer for reward inference
        #     import time

        #     t = time.strftime("%Y_%m_%d_%H_%M_%S")
        #     path = "logs/reward_inference"
        #     if self.log_file_path is not None:
        #         path = f"logs/{self.log_file_path}".replace("\\", "")

        #     os.makedirs(f"{path}/traj_state_buffer", exist_ok=True)
        #     with open(f"{path}/traj_state_buffer/buffer", "wb") as fp:
        #         pickle.dump(self.whole_state_buffer, fp)
        #     self.whole_state_buffer = []

        #     os.makedirs(f"{path}/traj_useful_recorded_data", exist_ok=True)
        #     with open(f"{path}/traj_useful_recorded_data/buffer", "wb") as fp2:
        #         pickle.dump(self.useful_recorded_data, fp2)
        #     self.useful_recorded_data = []

        return self._get_obs(), reward, done, False, {}

    def render(self):
        if self.visual or self.render_mode is not None:
            draw_state = self.vis.draw(self.sim)
            if self.visual_record:
                self.vis.record_frame()
            return draw_state

    def close(self):
        if self.visual_record:
            self.vis.close_recording()
