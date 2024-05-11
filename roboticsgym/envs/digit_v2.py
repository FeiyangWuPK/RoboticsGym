import datetime
import os
import numpy as np
import cv2

import transforms3d as tf3

import mujoco
import mujoco_viewer

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import time

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}

import numpy as np


#
# def x_vel_tracking(x_vel, command):
#     x_vel_error = x_vel - command
#     return np.exp(-np.square(x_vel_error))
#
# def y_vel_tracking(y_vel, command):
#     y_vel_error = y_vel - command
#     return np.exp(-0.1*np.square(y_vel_error))
#
#
# def ang_vel_tracking(ang_vel, command):
#     ang_vel_error = ang_vel - command  # yaw
#     return np.exp(-np.square(ang_vel_error))
#
#
# def base_motion(lin_vel, ang_vel):
#     # return np.square(lin_vel[2]) + 0.5 * np.sum(np.square(ang_vel[:2]))
#     return 0.2 * np.fabs(ang_vel[0]) + 0.2 * np.fabs(ang_vel[1]) + 0.8 * np.square(lin_vel[2])
#
#
# def base_orientation(gravity_vec):
#     return abs(gravity_vec[0])
#
#
# def torque_regularization(torque):
#     return np.sum(np.square(torque))

import enum


class StepType(enum.IntEnum):
    """Defines the status of a :class:`~TimeStep` within a sequence.

    Note that the last :class:`~TimeStep` in a sequence can either be
    :attribute:`StepType.TERMINAL` or :attribute:`StepType.TIMEOUT`.

    Suppose max_episode_length = 5:
    * A success sequence terminated at step 4 will look like:
        FIRST, MID, MID, TERMINAL
    * A success sequence terminated at step 5 will look like:
        FIRST, MID, MID, MID, TERMINAL
    * An unsuccessful sequence truncated by time limit will look like:
        FIRST, MID, MID, MID, TIMEOUT
    """

    # Denotes the first :class:`~TimeStep` in a sequence.
    FIRST = 0
    # Denotes any :class:`~TimeStep` in the middle of a sequence (i.e. not the
    # first or last one).
    MID = 1
    # Denotes the last :class:`~TimeStep` in a sequence that terminates
    # successfully.
    TERMINAL = 2
    # Denotes the last :class:`~TimeStep` in a sequence truncated by time
    # limit.
    TIMEOUT = 3

    @classmethod
    def get_step_type(cls, step_cnt, max_episode_length, done):
        """Determines the step type based on step cnt and done signal.

        Args:
            step_cnt (int): current step cnt of the environment.
            max_episode_length (int): maximum episode length.
            done (bool): the done signal returned by Environment.

        Returns:
            StepType: the step type.

        Raises:
            ValueError: if step_cnt is < 1. In this case a environment's
            `reset()` is likely not called yet and the step_cnt is None.
        """
        if max_episode_length is not None and step_cnt >= max_episode_length:
            return StepType.TIMEOUT
        elif done:
            return StepType.TERMINAL
        elif step_cnt == 1:
            return StepType.FIRST
        elif step_cnt < 1:
            raise ValueError(
                "Expect step_cnt to be >= 1, but got {} "
                "instead. Did you forget to call `reset("
                ")`?".format(step_cnt)
            )
        else:
            return StepType.MID


# reward for joint position difference
def joint_pos_tracking(CurrJoint_pos, RefJoint_pos):
    joint_pos_error = np.sum((CurrJoint_pos - RefJoint_pos) ** 2)
    return np.exp(-5 * joint_pos_error)


# reward for joint velocity difference
def joint_vel_tracking(CurrJoint_vel, RefJoint_vel):
    joint_pos_error = np.sum((CurrJoint_vel - RefJoint_vel) ** 2)
    return np.exp(-0.1 * joint_pos_error)


# reward for pelvis position difference
def root_pos_tracking(Currbase_pos, Refbase_pos):
    position_error = np.sum((Currbase_pos[:3] - Refbase_pos[:3]) ** 2)
    rotation_error = np.sum((Currbase_pos[3:] - Refbase_pos[3:]) ** 2)
    return np.exp(-20 * position_error - 10 * rotation_error)


# reward for pelvis orientation difference
def root_ori_tracking(Currbase_ori, Refbase_ori):
    linear_vel_error = np.sum((Currbase_ori[:3] - Refbase_ori[:3]) ** 2)
    angular_vel_error = np.sum((Currbase_ori[3:] - Refbase_ori[3:]) ** 2)
    return np.exp(-2 * linear_vel_error - 0.2 * angular_vel_error)


def lin_vel_tracking(lin_vel, command):
    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = np.sum(np.square(command[:2] - lin_vel[:2]))
    return np.exp(-lin_vel_error / 0.25)


def ang_vel_tracking(ang_vel, command):
    # Tracking of angular velocity commands (yaw)
    ang_vel_error = np.square(command[2] - ang_vel[2])
    return np.exp(-ang_vel_error / 0.25)


def z_vel_penalty(lin_vel):
    # Penalize z axis base linear velocity
    return np.square(lin_vel[2])


def roll_pitch_penalty(ang_vel):
    # Penalize xy axes base angular velocity
    return np.sum(np.square(ang_vel[:2]))


def base_orientation_penalty(projected_gravity):
    # Penalize non flat base orientation
    return np.sum(np.square(projected_gravity[:2]))


def torque_penalty(torque):
    return np.sum(np.square(torque))


def foot_lateral_distance_penalty(
    rfoot_poses, lfoot_poses
):  # TODO: check if this is correct
    """
    Get the closest distance between the two feet and make it into a penalty. The given points are five key points in the feet.
    Args:
        rfoot_poses: [3,5]
        lfoot_poses: [3,5]
    """
    assert rfoot_poses.shape == (3, 5) and lfoot_poses.shape == (
        3,
        5,
    ), "foot poses should be 5x3"

    distance0 = np.abs(rfoot_poses[1, 0] - lfoot_poses[1, 0])
    distance1 = np.abs(rfoot_poses[1, 4] - lfoot_poses[1, 3])
    distance2 = np.abs(rfoot_poses[1, 3] - lfoot_poses[1, 4])
    distances = np.array([distance0, distance1, distance2])
    closest_distance = np.min(distances)

    # return (closest_distance<0.27) * closest_distance
    return closest_distance < 0.13


def swing_foot_fix_penalty(lfoot_grf, rfoot_grf, action):
    """penalize if the toe joint changes from its fixed position in swing phase"""
    # TODO: check if contact check is correct
    lfoot_penalty = (lfoot_grf < 1) * np.sum(np.square(action[4:6]))
    rfoot_penalty = (rfoot_grf < 1) * np.sum(np.square(action[10:12]))
    return lfoot_penalty + rfoot_penalty


class DigitEnvBase(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 33,
    }

    def __init__(self, cfg, log_dir=""):

        self.frame_skip = 10
        dir_path, name = os.path.split(os.path.abspath(__file__))

        MujocoEnv.__init__(
            self,
            os.path.join(dir_path, "../../models/flat/digit-v3-flat.xml"),
            self.frame_skip,
            observation_space=Box(
                low=-np.inf, high=np.inf, shape=(106,), dtype=np.float64
            ),
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode="human",
        )

        self.action_space = Box(low=-1, high=1, shape=(20,), dtype=np.float32)

        self.home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
        self._env_id = None  # this should be set latter in training
        self.log_dir = log_dir

        # self.log_data = np.empty((0,13))
        self.log_data = np.empty((0, 169))
        self.log_ref_data = np.empty((0, 61 + 54 + 54))
        self.log_time = []
        self.log_data_flag = False

        self.mujoco2ar_index = [
            0,
            1,
            2,
            3,
            4,
            5,
            10,
            11,
            12,
            13,
            14,
            15,
            6,
            7,
            8,
            9,
            16,
            17,
            18,
            19,
        ]
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

        # self.ref_traj_dir = os.path.join(self.home_path, 'logs/record_vel_track/ref_traj_marching_3s')
        self.ref_traj_dir = os.path.join(
            self.home_path, "logs/record_vel_track/crocoddyl_ref_traj_taichi_3s"
        )

        self.ref_data = np.loadtxt(
            self.ref_traj_dir, delimiter=",", dtype="str", comments=None
        )
        self.ref_data = self.ref_data[:, 1:].astype(float)  # data_without_time
        self.ref_qpos = self.ref_data[:, 0:61]
        self.ref_qvel = self.ref_data[:, 61:115]
        # self.ref_qacc = self.ref_data[:, 115:169]
        self.ref_motion_len = len(self.ref_data)

        self.ref_a_pos = self.ref_qpos[:, self.p_index]  # actuation reference position
        self.ref_a_vel = self.ref_qvel[:, self.v_index]
        # self.ref_a_acc = self.ref_qacc[:, self.v_index]

        # config
        self.cfg = cfg

        # constants
        self.max_episode_length = int(
            self.cfg.env.max_time / self.cfg.control.control_dt
        )
        self.num_substeps = (
            int(self.cfg.control.control_dt / (self.cfg.env.sim_dt + 1e-8)) + 1
        )
        self.record_interval = int(
            1 / (self.cfg.control.control_dt * self.cfg.vis_record.record_fps)
        )

        self.history_len = int(self.cfg.env.hist_len_s / self.cfg.control.control_dt)
        self.hist_interval = int(
            self.cfg.env.hist_interval_s / self.cfg.control.control_dt
        )

        self.resampling_time = int(
            self.cfg.commands.resampling_time / self.cfg.control.control_dt
        )

        # control constants that can be changed with DR
        self.kp = self.cfg.control.default_kp
        self.kd = self.cfg.control.default_kd
        self.default_geom_friction = None
        self.motor_joint_friction = np.zeros(20)

        # containers (should be reset in reset())
        self.action = None  # only lower body
        self.full_action = None  # full body
        self.usr_command = None
        self._step_cnt = None
        self.max_traveled_distance = None

        self.joint_pos_hist = None
        self.joint_vel_hist = None
        # assert self.cfg.env.obs_dim - 70 == int(self.history_len / self.hist_interval) * 12 * 2 # lower motor joints only

        # containers (should be set at the _post_physics_step())
        self._terminal = None  # NOTE: this is internal use only, for the outside terminal check, just use eps.last, eps.terminal, eps.timeout from "EnvStep" or use env_infos "done"
        self.step_type = None

        # containters (should be set at the _get_obs())
        self.actor_obs = None
        self.value_obs = None
        self.robot_state = None

        # containers (should be initialized in child classes)
        self._interface = None
        self.nominal_qvel = None
        self.nominal_qpos = None
        self.nominal_motor_offset = None
        self.model = None
        self.data = None
        self._mbc = None

        self.curr_terrain_level = None

    def reset(self, seed=None):
        # domain randomization
        if self.cfg.domain_randomization.is_true:
            self.domain_randomization()
        # TODO: debug everything here
        # reset containers
        self._step_cnt = 0

        self.max_traveled_distance = 0.0

        # reset containers that are used in _get_obs
        self.joint_pos_hist = [np.zeros(12)] * self.history_len
        self.joint_vel_hist = [np.zeros(12)] * self.history_len
        self.action = np.zeros(self.cfg.env.act_dim, dtype=np.float32)
        # self._sample_commands()

        # setstate for initialization
        self._reset_state()

        # observe for next step
        self._get_obs()  # call _reset_state and _sample_commands before this.

        # start rendering
        if self._viewer is not None and self.cfg.vis_record.visualize:
            frame = self.render()
            if frame is not None:
                self.frames.append(frame)

        # # reset mbc
        # self._do_extra_in_reset() # self.robot_state should be updated before this by calling _get_obs

        self._step_assertion_check()

        # log data at TIMEOUT
        if StepType.TIMEOUT:
            # Writing to CSV file

            # np.savetxt(self.log_dir, np.column_stack((self.log_time,self.log_data)), delimiter=',', fmt='%s', comments='')
            np.savetxt(
                self.log_dir,
                np.column_stack((self.log_time, self.log_ref_data)),
                delimiter=",",
                fmt="%s",
                comments="",
            )
            # print(f'Data has been stored in {self.log_dir}')

        # second return is for episodic info
        # not sure if copy is needed but to make sure...
        return self.get_eps_info()

    def step(self, action):

        st_time = time.time()

        # print('action',action)
        if self._step_cnt is None:
            raise RuntimeError("reset() must be called before step()!")

        hat_target_pos = self.ref_a_pos[self._step_cnt]
        hat_target_vel = self.ref_a_vel[self._step_cnt]
        # hat_target_acc = self.ref_a_acc[self._step_cnt]

        adjusted_target_pos = hat_target_pos + np.concatenate((action[:6], action[6:]))

        # # clip action
        # self.action = np.clip(action, self.action_space.low, self.action_space.high)
        # self.full_action = np.concatenate((self.action[:6], np.zeros(4), self.action[6:], np.zeros(4))) # action space is only leg. actual motor inlcudes upper body.
        start = time.time()
        # control step
        if self.cfg.control.control_type == "PD":
            target_joint = adjusted_target_pos
            # target_joint = self.full_action * self.cfg.control.action_scale + self.nominal_motor_offset
            # print(self.full_action * self.cfg.control.action_scale)
            if self.cfg.domain_randomization.is_true:
                self.action_delay_time = int(
                    np.random.uniform(0, self.cfg.domain_randomization.action_delay, 1)
                    / self.cfg.env.sim_dt
                )
            self._pd_control(target_joint, np.zeros_like(target_joint))
        if self.cfg.control.control_type == "T":
            self._torque_control(self.full_action)

        rewards, tot_reward = self._post_physics_step()

        info = {
            "time": time.time() - start,
            "reward_info": rewards,
            "tot_reward": tot_reward,
            "next_value_obs": self.value_obs.copy(),
            "curr_value_obs": None,
            "robot_state": self.robot_state.copy(),
            "done": self.step_type is StepType.TERMINAL
            or self.step_type is StepType.TIMEOUT,
        }

        observation = self.actor_obs.copy()  # this observation is next state

        end_time = time.time()
        # if (end_time - st_time) > self.cfg.control.control_dt:
        #     print("the simulation looks slower than it actually is")
        if (end_time - st_time) < self.cfg.control.control_dt:
            time.sleep(self.cfg.control.control_dt - (end_time - st_time))

        return (
            observation,
            tot_reward,
            self.step_type is StepType.TERMINAL or self.step_type is StepType.TIMEOUT,
            False,
            info,
        )

    def get_eps_info(self):
        """
        return current environment's info.
        These informations are used when starting the episodes. starting obeservations.
        """
        return self.actor_obs.copy(), dict(
            curr_value_obs=self.value_obs.copy(), robot_state=self.robot_state.copy()
        )

    """ 
    internal helper functions 
    """

    def _sample_commands(self):
        """
        sample command for env
        make sure to call mbc.set_usr_command or mbc.reset after this. so that mbc's usr command is sync with env's
        """
        # Random command sampling in reset
        usr_command = np.zeros(3, dtype=np.float32)
        usr_command[0] = np.random.uniform(
            self.cfg.commands.ranges.x_vel_range[0],
            self.cfg.commands.ranges.x_vel_range[1],
        )

        usr_command[1] = np.random.uniform(
            self.cfg.commands.ranges.y_vel_range[0],
            self.cfg.commands.ranges.y_vel_range[1],
        )
        usr_command[2] = np.random.uniform(
            self.cfg.commands.ranges.ang_vel_range[0],
            self.cfg.commands.ranges.ang_vel_range[1],
        )

        if abs(usr_command[0]) < self.cfg.commands.ranges.cut_off:
            usr_command[0] = 0.0
        if abs(usr_command[1]) < self.cfg.commands.ranges.cut_off:
            usr_command[1] = 0.0
        if abs(usr_command[2]) < self.cfg.commands.ranges.cut_off:
            usr_command[2] = 0.0
        self.usr_command = usr_command
        # print("usr_command: ", self.usr_command)

    def _set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        # mujoco.mj_forward(self.model, self.data)
        # print(self.data.qvel)

    def _reset_state(self):
        raise NotImplementedError

    def _torque_control(self, torque):
        ratio = self._interface.get_gear_ratios().copy()
        for _ in range(
            self._num_substeps
        ):  # this is open loop torque control. no feedback.
            tau = [
                (i / j) for i, j in zip(torque, ratio)
            ]  # TODO: why divide by ratio..? This need to be checked
            self._interface.set_motor_torque(tau)
            self._interface.step()

    def _pd_control(self, target_pos, target_vel):
        self._interface.set_pd_gains(self.kp, self.kd)
        ratio = self._interface.get_gear_ratios().copy()
        for cnt in range(self.num_substeps):  # this is PD feedback loop
            if self.cfg.domain_randomization.is_true:
                motor_vel = self._interface.get_act_joint_velocities()
                motor_joint_friction = self.motor_joint_friction * np.sign(motor_vel)
                if cnt < self.action_delay_time:
                    tau = motor_joint_friction
                    tau = [(i / j) for i, j in zip(tau, ratio)]
                    self._interface.set_motor_torque(tau)
                    self._interface.step()
                else:
                    tau = self._interface.step_pd(target_pos, target_vel)
                    tau += motor_joint_friction
                    tau = [(i / j) for i, j in zip(tau, ratio)]
                    self._interface.set_motor_torque(tau)
                    self._interface.step()
            else:
                tau = self._interface.step_pd(
                    target_pos, target_vel
                )  # this tau is joint space torque
                tau = [(i / j) for i, j in zip(tau, ratio)]
                self._interface.set_motor_torque(tau)
                self._interface.step()

    def _post_physics_step(self):

        # observe for next step

        self._get_obs()

        self._is_terminal()

        # TODO: debug reward function
        rewards, tot_reward = self._compute_reward()

        # visualize
        if (
            self._viewer is not None
            and self._step_cnt % self.record_interval == 0
            and self.cfg.vis_record.visualize
            and self._step_cnt
        ):
            frame = self.render()
            if frame is not None:
                self.frames.append(frame)

        self._step_cnt += 1
        self.step_type = StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self.max_episode_length,
            done=self._terminal,
        )
        if self.step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None  # this becomes zero when reset is called

        return rewards, tot_reward

    def _get_obs(self):
        """ "
        update actor_obs, value_obs, robot_state, all the other states
        make sure to call _reset_state and _sample_commands before this.
        self._mbc is not reset when first call but it is okay for self._mbc.get_phase_variable(), self._mbc.get_domain(). check those functions.
        """
        # TODO: check all the values
        # update statesself.log_data = np.vstack([ self.log_data, np.concatenate((self.qpos, self.qvel, self.qacc), axis=0) ])
        self.qpos = self.data.qpos.copy()
        self.qvel = self.data.qvel.copy()
        self.qacc = self.data.qacc.copy()

        self._update_root_state()
        self._update_joint_state()
        self._update_joint_hist()
        self._update_robot_state()

        # log data in CSV file
        if self.log_data_flag:
            self.log_time.append(str(datetime.datetime.now()))
            # self.log_data = np.vstack([ self.log_data, np.append(self.qpos[:7], self.qvel[:6]) ])
            self.log_ref_data = np.vstack(
                [
                    self.log_ref_data,
                    np.concatenate((self.qpos, self.qvel, self.qacc), axis=0),
                ]
            )

        # update observations
        self.projected_gravity = self._interface.get_projected_gravity_vec()
        self.noisy_projected_gravity = self.projected_gravity + np.random.normal(
            0, self.cfg.obs_noise.projected_gravity_std, 3
        )
        self.noisy_projected_gravity = self.noisy_projected_gravity / np.linalg.norm(
            self.noisy_projected_gravity
        )
        self._update_actor_obs()
        # self.value_obs = self.actor_obs.copy() # TODO: apply dreamwaq
        self._update_critic_obs()

        # update traveled distance
        self.max_traveled_distance = max(
            self.max_traveled_distance, np.linalg.norm(self.root_xy_pos[:2])
        )

        # not sure if copy is needed but to make sure...

    def _update_root_state(self):
        # root states
        self.root_xy_pos = self.qpos[0:2]
        self.root_world_height = self.qpos[2]
        self.root_quat = self.qpos[3:7]
        roll, pitch, yaw = tf3.euler.quat2euler(self.root_quat, axes="sxyz")
        base_rot = tf3.euler.euler2mat(0, 0, yaw, "sxyz")
        self.root_lin_vel = np.transpose(base_rot).dot(self.qvel[0:3])
        self.root_ang_vel = np.transpose(base_rot).dot(self.qvel[3:6])
        self.noisy_root_ang_vel = self.root_ang_vel + np.random.normal(
            0, self.cfg.obs_noise.ang_vel_std, 3
        )
        self.noisy_root_lin_vel = self.root_lin_vel + np.random.normal(
            0, self.cfg.obs_noise.lin_vel_std, 3
        )

    def _update_joint_state(self):
        # motor states
        self.motor_pos = self._interface.get_act_joint_positions()
        self.motor_vel = self._interface.get_act_joint_velocities()
        # passive hinge states
        self.passive_hinge_pos = self._interface.get_passive_hinge_positions()
        self.passive_hinge_vel = self._interface.get_passive_hinge_velocities()

        self.noisy_motor_pos = self.motor_pos + np.random.normal(
            0, self.cfg.obs_noise.dof_pos_std, 20
        )
        self.noisy_motor_vel = self.motor_vel + np.random.normal(
            0, self.cfg.obs_noise.dof_vel_std, 20
        )
        self.noisy_passive_hinge_pos = self.passive_hinge_pos + np.random.normal(
            0, self.cfg.obs_noise.dof_pos_std, 10
        )
        self.noisy_passive_hinge_vel = self.passive_hinge_vel + np.random.normal(
            0, self.cfg.obs_noise.dof_vel_std, 10
        )

    def _update_joint_hist(self):
        # joint his buffer update
        self.joint_pos_hist.pop(0)
        self.joint_vel_hist.pop(0)
        if self.cfg.obs_noise.is_true:
            self.joint_pos_hist.append(
                np.array(self.noisy_motor_pos)[self.cfg.control.lower_motor_index]
            )
            self.joint_vel_hist.append(
                np.array(self.noisy_motor_vel)[self.cfg.control.lower_motor_index]
            )
        else:
            self.joint_pos_hist.append(
                np.array(self.motor_pos)[self.cfg.control.lower_motor_index]
            )
            self.joint_vel_hist.append(
                np.array(self.motor_vel)[self.cfg.control.lower_motor_index]
            )
        assert len(self.joint_vel_hist) == self.history_len

        # assign joint history obs
        self.joint_pos_hist_obs = []
        self.joint_vel_hist_obs = []
        for i in range(int(self.history_len / self.hist_interval)):
            self.joint_pos_hist_obs.append(self.joint_pos_hist[i * self.hist_interval])
            self.joint_vel_hist_obs.append(self.joint_vel_hist[i * self.hist_interval])
        assert len(self.joint_pos_hist_obs) == 3
        self.joint_pos_hist_obs = np.concatenate(self.joint_pos_hist_obs).flatten()
        self.joint_vel_hist_obs = np.concatenate(self.joint_vel_hist_obs).flatten()

    def _update_robot_state(self):
        """robot state is state used for MBC"""
        # body_height = self.root_world_height - self._get_height(self.root_xy_pos[0] , self.root_xy_pos[1])
        body_height = self.root_world_height
        root_pos = np.array([self.root_xy_pos[0], self.root_xy_pos[1], body_height])
        self.robot_state = np.concatenate(
            [
                root_pos,  # 2 0~3
                self.root_quat,  # 4 3~7
                self.root_lin_vel,  # 3 7~10
                self.root_ang_vel,  # 3 10~13
                self.motor_pos,  # 20 13~33
                self.passive_hinge_pos,  # 10 33~43
                self.motor_vel,  # 20     43~63
                self.passive_hinge_vel,  # 10 63~73
            ]
        )

    # def _update_actor_obs(self):
    #     # NOTE: make sure to call get_action from self._mbc so that phase_variable is updated
    #     if self.cfg.obs_noise.is_true:
    #         self.actor_obs = np.concatenate([self.noisy_root_lin_vel * self.cfg.normalization.obs_scales.lin_vel, # 3
    #                                          self.noisy_root_ang_vel * self.cfg.normalization.obs_scales.ang_vel, # 3
    #                                          self.noisy_projected_gravity, # 3
    #                                          self.usr_command * [self.cfg.normalization.obs_scales.lin_vel,
    #                                                              self.cfg.normalization.obs_scales.lin_vel,
    #                                                              self.cfg.normalization.obs_scales.ang_vel], # 3
    #                                         np.array(self.noisy_motor_pos)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_pos, # 12
    #                                         np.array(self.noisy_passive_hinge_pos) * self.cfg.normalization.obs_scales.dof_pos, # 10
    #                                         np.array(self.noisy_motor_vel)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_vel, # 12
    #                                         np.array(self.noisy_passive_hinge_vel) * self.cfg.normalization.obs_scales.dof_vel, # 10
    #                                         np.array([self._mbc.get_phase_variable(), self._mbc.get_domain()]), # 2
    #                                         self.action.copy(), # 12
    #                                         self.joint_pos_hist_obs * self.cfg.normalization.obs_scales.dof_pos,
    #                                         self.joint_vel_hist_obs * self.cfg.normalization.obs_scales.dof_vel]).astype(np.float32).flatten()
    #     else:
    #         self.actor_obs = np.concatenate([self.root_lin_vel * self.cfg.normalization.obs_scales.lin_vel, # 3
    #                                          self.root_ang_vel * self.cfg.normalization.obs_scales.ang_vel, # 3
    #                                          self.projected_gravity, # 3
    #                                          self.usr_command * [self.cfg.normalization.obs_scales.lin_vel,
    #                                                              self.cfg.normalization.obs_scales.lin_vel,
    #                                                              self.cfg.normalization.obs_scales.ang_vel], # 3
    #                                         np.array(self.motor_pos)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_pos, # 12
    #                                         np.array(self.passive_hinge_pos) * self.cfg.normalization.obs_scales.dof_pos, # 10
    #                                         np.array(self.motor_vel)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_vel, # 12
    #                                         np.array(self.passive_hinge_vel) * self.cfg.normalization.obs_scales.dof_vel, # 10
    #                                         np.array([self._mbc.get_phase_variable(), self._mbc.get_domain()]), # 2
    #                                         self.action.copy(), # 12
    #                                         self.joint_pos_hist_obs * self.cfg.normalization.obs_scales.dof_pos,
    #                                         self.joint_vel_hist_obs * self.cfg.normalization.obs_scales.dof_vel]).astype(np.float32).flatten()

    #     assert self.actor_obs.shape[0] == self.cfg.env.obs_dim

    def _update_actor_obs(self):
        # NOTE: make sure to call get_action from self._mbc so that phase_variable is updated
        if self.cfg.obs_noise.is_true:
            self.actor_obs = (
                np.concatenate(
                    [
                        self.noisy_root_lin_vel,  # 3
                        self.noisy_root_ang_vel,  # 3
                        #  self.noisy_projected_gravity, # 3
                        self.qpos[:7],
                        self.qpos[self.p_index],
                        self.qvel[self.v_index],  # 47
                        self.ref_qpos[self._step_cnt, :7],
                        self.ref_qvel[self._step_cnt, :6],
                        self.ref_a_pos[self._step_cnt],
                        self.ref_a_vel[self._step_cnt],  # 53
                        # self.action.copy(), # 12
                        # self.joint_pos_hist_obs,
                        # self.joint_vel_hist_obs,
                    ]
                )
                .astype(np.float32)
                .flatten()
            )
        else:
            self.actor_obs = (
                np.concatenate(
                    [
                        self.root_lin_vel,  # 3
                        self.root_ang_vel,  # 3
                        # self.projected_gravity, # 3
                        self.qpos[:7],  # 7
                        self.qpos[self.p_index],  # 20
                        self.qvel[self.v_index],  # 20
                        self.ref_qpos[self._step_cnt, :7],
                        self.ref_qvel[self._step_cnt, :6],
                        self.ref_a_pos[self._step_cnt],
                        self.ref_a_vel[self._step_cnt],  # 53
                        # self.action.copy(), # 12
                        # self.joint_pos_hist_obs,
                        # self.joint_vel_hist_obs,
                    ]
                )
                .astype(np.float32)
                .flatten()
            )

        assert self.actor_obs.shape[0] == self.cfg.env.obs_dim

    # def _update_critic_obs(self):
    #     self.value_obs = np.concatenate([self.root_lin_vel * self.cfg.normalization.obs_scales.lin_vel, # 3
    #                                     self.root_ang_vel * self.cfg.normalization.obs_scales.ang_vel, # 3
    #                                     self.projected_gravity, # 3
    #                                     self.usr_command * [self.cfg.normalization.obs_scales.lin_vel,
    #                                                         self.cfg.normalization.obs_scales.lin_vel,
    #                                                         self.cfg.normalization.obs_scales.ang_vel], # 3
    #                                 np.array(self.motor_pos)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_pos, # 12
    #                                 np.array(self.passive_hinge_pos) * self.cfg.normalization.obs_scales.dof_pos, # 10
    #                                 np.array(self.motor_vel)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_vel, # 12
    #                                 np.array(self.passive_hinge_vel) * self.cfg.normalization.obs_scales.dof_vel, # 10
    #                                 np.array([self._mbc.get_phase_variable(), self._mbc.get_domain()]), # 2
    #                                 self.action.copy(), # 12
    #                                 self.kp,
    #                                 self.kd,
    #                                 self.motor_joint_friction
    #                                 ]).astype(np.float32).flatten()

    #     assert self.value_obs.shape[0] == self.cfg.env.value_obs_dim

    def _update_critic_obs(self):
        self.value_obs = (
            np.concatenate(
                [
                    self.root_lin_vel * self.cfg.normalization.obs_scales.lin_vel,  # 3
                    self.root_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                    # self.projected_gravity, # 3
                    self.qpos[:7],
                    self.qpos[self.p_index],
                    self.qvel[self.v_index],
                    self.ref_qpos[self._step_cnt, :7],
                    self.ref_qvel[self._step_cnt, :6],
                    self.ref_a_pos[self._step_cnt],
                    self.ref_a_vel[self._step_cnt],  # 53
                    # self.action.copy(), # 12
                    # self.kp,
                    # self.kd,
                ]
            )
            .astype(np.float32)
            .flatten()
        )

        assert self.value_obs.shape[0] == self.cfg.env.value_obs_dim

    def _do_extra_in_reset(self):
        self._mbc.reset(
            self.robot_state, self.usr_command
        )  # get_obs should be called before this to update robot_state

    def _step_assertion_check(self):
        assert self._mbc.get_phase_variable() == 0.0
        assert self._mbc.get_domain() == 0
        # assert self.usr_command is not None
        # assert self._mbc.usr_command is not None

    def _is_terminal(self):
        # self_collision_check = self._interface.check_self_collisions()
        # bad_collision_check = self._interface.check_bad_collisions()
        # lean_check = self._interface.check_body_lean()  # TODO: no lean when RL training. why...?
        # terminate_conditions = {"self_collision_check": self_collision_check,
        #                         "bad_collision_check": bad_collision_check,
        #                         # "body_lean_check": lean_check,
        #                         }

        root_vel_crazy_check = (
            (self.root_lin_vel[0] > 1.5)
            or (self.root_lin_vel[1] > 1.5)
            or (self.root_lin_vel[2] > 1.0)
        )  # as in digit controller
        self_collision_check = self._interface.check_self_collisions()
        body_lean_check = self._interface.check_body_lean()
        ref_traj_step_check = self._step_cnt > self.ref_motion_len - 3
        # mbc_divergence_check = np.isnan(self._mbc_torque).any() or np.isnan(self._mbc_action).any() #TODO:remove this when RL.
        terminate_conditions = {
            "root_vel_crazy_check": root_vel_crazy_check,
            "self_collision_check": self_collision_check,
            "body_lean_check": body_lean_check,
            "ref_traj_step_check": ref_traj_step_check,
            # "mbc_divergence_check": mbc_divergence_check
        }

        self._terminal = True in terminate_conditions.values()

    def _compute_reward(self):
        # the states are after stepping.

        joint_pos_tracking_reward = joint_pos_tracking(
            self.qpos[7:], self.ref_qpos[self._step_cnt, 7:]
        )
        joint_vel_tracking_reward = joint_vel_tracking(
            self.qvel[6:], self.ref_qvel[self._step_cnt, 6:]
        )
        root_pos_tracking_reward = root_pos_tracking(
            self.qpos[:7], self.ref_qpos[self._step_cnt, :7]
        )
        root_ori_tracking_reward = root_ori_tracking(
            self.qvel[:6], self.ref_qvel[self._step_cnt, :6]
        )

        # lin_vel_tracking_reward = lin_vel_tracking(self.root_lin_vel, self.usr_command)
        # ang_vel_tracking_reward = ang_vel_tracking(self.root_ang_vel, self.usr_command)
        # z_vel_penalty_reward = z_vel_penalty(self.root_lin_vel)
        # roll_pitch_penalty_reward = roll_pitch_penalty(self.root_ang_vel)
        # base_orientation_penalty_reward = base_orientation_penalty(self.projected_gravity)
        # torque = np.array(self._interface.get_act_joint_torques())[self.cfg.control.lower_motor_index]
        # torque_penalty_reward = torque_penalty(torque)

        # rfoot_pose = np.array(self._interface.get_rfoot_keypoint_pos()).T
        # lfoot_pose = np.array(self._interface.get_lfoot_keypoint_pos()).T
        # rfoot_pose = self._interface.change_positions_to_rotated_world_frame(rfoot_pose)
        # lfoot_pose = self._interface.change_positions_to_rotated_world_frame(lfoot_pose)

        # foot_lateral_distance_penalty_reward = 1.0 if foot_lateral_distance_penalty(rfoot_pose, lfoot_pose) else 0.

        # rfoot_grf = self._interface.get_rfoot_grf()
        # lfoot_grf = self._interface.get_lfoot_grf()

        # swing_foot_fix_penalty_reward =  swing_foot_fix_penalty(lfoot_grf, rfoot_grf, self.action)

        termination_reward = -1.0 if self._terminal else 0.0

        rewards_tmp = {
            "joint_pos_tracking": joint_pos_tracking_reward,
            "joint_vel_tracking": joint_vel_tracking_reward,
            "root_pos_tracking": root_pos_tracking_reward,
            "root_ori_tracking": root_ori_tracking_reward,
            #    "lin_vel_tracking": lin_vel_tracking_reward,
            #    "ang_vel_tracking": ang_vel_tracking_reward,
            #    "z_vel_penalty": z_vel_penalty_reward,
            #    "roll_pitch_penalty": roll_pitch_penalty_reward,
            #    "base_orientation_penalty": base_orientation_penalty_reward,
            #    "torque_penalty": torque_penalty_reward,
            #    "foot_lateral_distance_penalty": foot_lateral_distance_penalty_reward,
            #    "swing_foot_fix_penalty": swing_foot_fix_penalty_reward,
            #    "termination": termination_reward,
        }
        rewards = {}
        tot_reward = 0.0
        for key in rewards_tmp.keys():
            rewards[key] = getattr(self.cfg.rewards.scales, key) * rewards_tmp[key]
            tot_reward += rewards[key]

        return rewards, tot_reward

    """
    Visualization Code
    """

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def visualize(self):
        """Creates a visualization of the environment."""
        assert self.cfg.vis_record.visualize, "you should set visualize flag to true"
        assert self._viewer is None, "there is another viewer"
        # if self._viewer is not None:
        #     #     self._viewer.close()
        #     #     self._viewer = None
        #     return
        if self.cfg.vis_record.record:
            self._viewer = mujoco_viewer.MujocoViewer(
                self.model, self.data, "offscreen"
            )
        else:
            self._viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer_setup()

    def viewer_setup(self):
        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._viewer.cam.fixedcamid = 0
        self._viewer.cam.distance = self.model.stat.extent * 1.5
        self._viewer.cam.lookat[2] = 0.0
        self._viewer.cam.lookat[0] = 0
        self._viewer.cam.lookat[1] = 0.0
        self._viewer.cam.azimuth = 180
        self._viewer.cam.distance = 5
        self._viewer.cam.elevation = -10
        self._viewer.vopt.geomgroup[0] = 1
        self._viewer._render_every_frame = True
        # self.viewer._run_speed *= 20
        self._viewer._contacts = True
        self._viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = (
            self._viewer._contacts
        )
        self._viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = (
            self._viewer._contacts
        )

    def viewer_is_paused(self):
        return self._viewer._paused

    def render(self):
        assert self._viewer is not None
        if self.cfg.vis_record.record:
            return self._viewer.read_pixels(camid=0)
        else:
            self._viewer.render()
            return None

    def save_video(self, name):
        assert self.cfg.vis_record.record
        assert self.log_dir is not None
        assert self._viewer is not None
        home_path = os.path.dirname(os.path.realpath(__file__)) + "/.."
        video_dir = os.path.join(
            os.path.join(home_path, "logs/record_vel_track/", "video")
        )
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, name + ".mp4")
        command_path = os.path.join(video_dir, name + ".txt")
        f = open(command_path, "w")
        f.write(str(self.usr_command))
        f.close()
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            self._record_fps,
            (self.frames[0].shape[1], self.frames[0].shape[0]),
        )
        for frame in self.frames:
            video_writer.write(frame)
        video_writer.release()
        self.clear_frames()

    def clear_frames(self):
        self.frames = []

    def domain_randomization(self):
        # NOTE: the parameters in mjModel shouldn't be changed in runtime!
        # self.model.geom_friction[:,0] = self.default_geom_friction[:,0] * np.random.uniform(self.cfg.domain_randomization.friction_noise[0],
        #                                                                           self.cfg.domain_randomization.friction_noise[1],
        #                                                                           size=self.default_geom_friction[:,0].shape)
        self.motor_joint_friction = np.random.uniform(
            self.cfg.domain_randomization.joint_friction[0],
            self.cfg.domain_randomization.joint_friction[1],
            size=self.motor_joint_friction.shape,
        )
        self.kp = self.cfg.control.default_kp * np.random.uniform(
            self.cfg.domain_randomization.kp_noise[0],
            self.cfg.domain_randomization.kp_noise[1],
            size=self.cfg.control.default_kp.shape,
        )
        self.kd = self.cfg.control.default_kd * np.random.uniform(
            self.cfg.domain_randomization.kd_noise[0],
            self.cfg.domain_randomization.kd_noise[1],
            size=self.cfg.control.default_kd.shape,
        )
