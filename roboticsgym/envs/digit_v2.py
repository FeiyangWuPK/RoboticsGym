import datetime
import os
import numpy as np
import cv2
import enum

import transforms3d as tf3

import mujoco
import mujoco_viewer

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import time

from roboticsgym.envs.reference_trajectories.loadDigit import DigitTrajectory
from roboticsgym.envs.cfg.digit_env_config import DigitEnvConfig


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}


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
    Get the closest distance between the two feet and make it into a penalty.
    The given points are five key points in the feet.
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


class DigitEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 200,
    }

    def __init__(self, cfg=DigitEnvConfig(), log_dir=""):

        self.frame_skip = 10
        dir_path, name = os.path.split(os.path.abspath(__file__))

        MujocoEnv.__init__(
            self,
            os.getcwd() + "/roboticsgym/envs/xml/digit.xml",
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

        # self.ref_traj_dir = os.path.join(
        #     self.home_path, "logs/record_vel_track/ref_traj_marching_3s"
        # )

        self.ref_trajectory = DigitTrajectory(
            os.getcwd()
            + "/roboticsgym/envs/"
            + "reference_trajectories/digit_state_20240422.csv"
        )
        # self.ref_traj_dir = os.path.join(
        #     self.home_path, "logs/record_vel_track/crocoddyl_ref_traj_taichi_3s"
        # )

        # self.ref_data = np.loadtxt(
        #     self.ref_traj_dir, delimiter=",", dtype="str", comments=None
        # )
        # self.ref_data = self.ref_data[:, 1:].astype(float)  # data_without_time
        # self.ref_qpos = self.ref_data[:, 0:61]
        # self.ref_qvel = self.ref_data[:, 61:115]
        # # self.ref_qacc = self.ref_data[:, 115:169]
        # self.ref_motion_len = len(self.ref_data)

        # self.ref_a_pos = self.ref_qpos[:, self.p_index]  # actuation reference position
        # self.ref_a_vel = self.ref_qvel[:, self.v_index]
        # # self.ref_a_acc = self.ref_qacc[:, self.v_index]
        self.ref_qpos = self.ref_trajectory.qpos[11000:]
        self.ref_qvel = self.ref_trajectory.qvel[11000:]
        self.ref_a_pos = self.ref_qpos[11000:, self.p_index]
        self.ref_a_vel = self.ref_qvel[11000:, self.v_index]
        self.ref_motion_len = self.ref_qpos.shape[0]

        self.ref_data = np.concatenate([self.ref_qpos, self.ref_qvel], axis=1)

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

        self._terminal = None
        # NOTE: this is internal use only,
        # for the outside terminal check, just use eps.last, eps.terminal,
        # eps.timeout from "EnvStep" or use env_infos "done"
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
        # class that have functions to get and set lowlevel mujoco simulation parameters
        self._interface = RobotInterface(
            self.model,
            self.data,
            "right-toe-roll",
            "left-toe-roll",
            "right-foot",
            "left-foot",
        )
        self.nominal_qvel = self.data.qvel.ravel().copy()
        self.nominal_qpos = self.model.keyframe("standing").qpos
        self.nominal_motor_offset = self.nominal_qpos[
            self._interface.get_motor_qposadr()
        ]

        self._record_fps = 15

        # setup viewer
        self.frames = []  # this only be cleaned at the save_video function
        self._viewer = None
        if self.cfg.vis_record.visualize:
            self.visualize()

        # defualt geom friction
        self.default_geom_friction = self.model.geom_friction.copy()
        # pickling
        kwargs = {
            "cfg": self.cfg,
            "log_dir": self.log_dir,
        }
        utils.EzPickle.__init__(self, **kwargs)

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

        # log data at TIMEOUT
        if StepType.TIMEOUT:
            # Writing to CSV file

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
        # hat_target_vel = self.ref_a_vel[self._step_cnt]
        # hat_target_acc = self.ref_a_acc[self._step_cnt]

        adjusted_target_pos = hat_target_pos + np.concatenate((action[:6], action[6:]))

        start = time.time()
        # control step
        if self.cfg.control.control_type == "PD":
            target_joint = adjusted_target_pos

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

    def _sample_commands(self):
        """
        sample command for env

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

    def _reset_state(self):
        init_qpos = self.nominal_qpos.copy()
        init_qvel = self.nominal_qvel.copy()
        init_qpos[0:2] = self.env_origin[:2]

        # dof randomized initialization
        if self.cfg.reset_state.random_dof_reset:
            init_qvel[:6] = init_qvel[:6] + np.random.normal(
                0, self.cfg.reset_state.root_v_std, 6
            )
            for joint_name in self.cfg.reset_state.random_dof_names:
                qposadr = self._interface.get_jnt_qposadr_by_name(joint_name)
                qveladr = self._interface.get_jnt_qveladr_by_name(joint_name)
                init_qpos[qposadr[0]] = init_qpos[qposadr[0]] + np.random.normal(
                    0, self.cfg.reset_state.p_std
                )
                init_qvel[qveladr[0]] = init_qvel[qveladr[0]] + np.random.normal(
                    0, self.cfg.reset_state.v_std
                )

        self._set_state(np.asarray(init_qpos), np.asarray(init_qvel))

        # adjust so that no penetration
        rfoot_poses = np.array(self._interface.get_rfoot_keypoint_pos())
        lfoot_poses = np.array(self._interface.get_lfoot_keypoint_pos())
        rfoot_poses = np.array(rfoot_poses)
        lfoot_poses = np.array(lfoot_poses)

        delta = np.max(
            np.concatenate([0.0 - rfoot_poses[:, 2], 0.0 - lfoot_poses[:, 2]])
        )
        init_qpos[2] = init_qpos[2] + delta + 0.02

        self._set_state(np.asarray(init_qpos), np.asarray(init_qvel))

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

    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        # hfield
        if hfieldid is not None:
            mujoco.mjr_uploadHField(self.model, self._viewer.ctx, hfieldid)
        # mesh
        if meshid is not None:
            mujoco.mjr_uploadMesh(self.model, self._viewer.ctx, meshid)
        # texture
        if texid is not None:
            mujoco.mjr_uploadTexture(self.model, self._viewer.ctx, texid)

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

        self._step_cnt += int(self.frame_skip / 2)
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

        """
        # TODO: check all the values

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

    def _update_actor_obs(self):
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

        terminate_conditions = {
            "root_vel_crazy_check": root_vel_crazy_check,
            "self_collision_check": self_collision_check,
            "body_lean_check": body_lean_check,
            "ref_traj_step_check": ref_traj_step_check,
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

        # termination_reward = -1.0 if self._terminal else 0.0

        rewards_tmp = {
            "joint_pos_tracking": joint_pos_tracking_reward,
            "joint_vel_tracking": joint_vel_tracking_reward,
            "root_pos_tracking": root_pos_tracking_reward,
            "root_ori_tracking": root_ori_tracking_reward,
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


class RobotInterface(object):
    def __init__(
        self,
        model,
        data,
        rfoot_body_name=None,
        lfoot_body_name=None,
        rfoot_collision_geom_name=None,
        lfoot_collision_geom_name=None,
    ):
        self.model = model
        self.data = data

        self.rfoot_body_name = rfoot_body_name
        self.lfoot_body_name = lfoot_body_name
        self.rfoot_collision_geom_name = rfoot_collision_geom_name
        self.lfoot_collision_geom_name = lfoot_collision_geom_name

        # the very first object(terrain) is named world.
        # https://mujoco.readthedocs.io/en/stable/modeling.html#kinematic-tree
        self.floor_body_name = "world"
        passive_hinge_names = [
            "left-shin",
            "left-tarsus",
            "left-toe-pitch",
            "left-toe-roll",
            "left-heel-spring",
            "right-shin",
            "right-tarsus",
            "right-toe-pitch",
            "right-toe-roll",
            "right-heel-spring",
        ]

        mujoco_passive_hinge_names = []  #
        for joint_name in self.get_joint_names():
            if joint_name in passive_hinge_names:
                mujoco_passive_hinge_names.append(joint_name)

        self.passive_hinge_inds_in_qpos = []
        for hinge_name in mujoco_passive_hinge_names:
            self.passive_hinge_inds_in_qpos.append(
                self.get_jnt_qposadr_by_name(hinge_name)[0]
            )

        self.passive_hinge_inds_in_qvel = []
        for hinge_name in mujoco_passive_hinge_names:
            self.passive_hinge_inds_in_qvel.append(
                self.get_jnt_qveladr_by_name(hinge_name)[0]
            )

    def get_passive_hinge_positions(self):
        joint_pos = self.get_qpos().copy()
        return joint_pos[self.passive_hinge_inds_in_qpos]

    def get_passive_hinge_velocities(self):
        joint_vel = self.get_qvel().copy()
        return joint_vel[self.passive_hinge_inds_in_qvel]

    def nq(self):
        return self.model.nq

    def nu(self):
        return self.model.nu

    def nv(self):
        return self.model.nv

    def sim_dt(self):
        return self.model.opt.timestep

    def get_robot_mass(self):
        return mujoco.mj_getTotalmass(self.model)

    def get_qpos(self):
        return self.data.qpos.copy()

    def get_qvel(self):
        return self.data.qvel.copy()

    def get_qacc(self):
        return self.data.qacc.copy()

    def get_cvel(self):
        return self.data.cvel.copy()

    def get_jnt_id_by_name(self, name):
        return self.model.joint(name)

    def get_jnt_qposadr_by_name(self, name):
        return self.model.joint(name).qposadr

    def get_jnt_qveladr_by_name(self, name):
        return self.model.joint(name).dofadr

    def get_body_ext_force(self):
        return self.data.cfrc_ext.copy()

    def get_motor_speed_limits(self):
        """
        Returns speed limits of the *actuator* in radians per sec.
        This assumes the actuator 'user' element defines speed limits
        at the actuator level in revolutions per minute.
        """
        rpm_limits = self.model.actuator_user[:, 0]  # RPM
        return (rpm_limits * (2 * np.pi / 60)).tolist()  # radians per sec

    def get_act_joint_speed_limits(self):
        """
        Returns speed limits of the *joint* in radians per sec.
        This assumes the actuator 'user' element defines speed limits
        at the actuator level in revolutions per minute.
        """
        gear_ratios = self.model.actuator_gear[:, 0]
        mot_lims = self.get_motor_speed_limits()
        return [float(i / j) for i, j in zip(mot_lims, gear_ratios)]

    def get_gear_ratios(self):
        """
        Returns transmission ratios.
        """
        return self.model.actuator_gear[:, 0]

    def get_motor_names(self):
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        return actuator_names

    def get_actuated_joint_inds(self):
        """
        Returns list of joint indices to which actuators are attached.
        """
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        # joint_names.pop(0) # remove none
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        return [idx for idx, jnt in enumerate(joint_names) if jnt in actuator_names]

    def get_actuated_joint_names(self):
        """
        Returns list of joint names to which actuators are attached.
        """
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        return [jnt for idx, jnt in enumerate(joint_names) if jnt in actuator_names]

    def get_joint_names(self):
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        joint_names.pop(0)
        return joint_names

    def get_motor_qposadr(self):
        """
        Returns the list of qpos indices of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()
        return [self.model.jnt_qposadr[i] for i in indices]

    def get_motor_positions(self):
        """
        Returns position of actuators.
        length means joint angle in case of hinge joint.
        It must be used with get_act_joint_positions.
        """
        return self.data.actuator_length.tolist()

    def get_motor_velocities(self):
        """
        Returns velocities of actuators. It must be used with get_act_joint_velocities.
        """
        return self.data.actuator_velocity.tolist()

    def get_act_joint_torques(self):
        """
        Returns actuator force in joint space.
        """
        gear_ratios = self.model.actuator_gear[:, 0]
        motor_torques = self.data.actuator_force.tolist()
        return [float(i * j) for i, j in zip(motor_torques, gear_ratios)]

    def get_act_joint_positions(self):
        """
        Returns position of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:, 0]
        motor_positions = self.get_motor_positions()
        return [float(i / j) for i, j in zip(motor_positions, gear_ratios)]

    def get_act_joint_velocities(self):
        """
        Returns velocities of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:, 0]
        motor_velocities = self.get_motor_velocities()
        return [float(i / j) for i, j in zip(motor_velocities, gear_ratios)]

    def get_act_joint_range(self):
        """
        Returns the lower and upper limits of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()
        low, high = self.model.jnt_range[indices, :].T
        return low, high

    def get_actuator_ctrl_range(self):
        """
        Returns the acutator ctrlrange defined in model xml.
        """
        low, high = self.model.actuator_ctrlrange.copy().T
        return low, high

    def get_actuator_user_data(self):
        """
        Returns the user data (if any) attached to each actuator.
        """
        return self.model.actuator_user.copy()

    def get_root_body_pos(self):
        return self.data.xpos[1].copy()

    def get_root_body_vel(self):
        qveladr = self.get_jnt_qveladr_by_name("base")
        return self.data.qvel[qveladr:qveladr+6].copy()

    def get_sensordata(self, sensor_name):
        sensor_id = self.model.sensor(sensor_name)
        sensor_adr = self.model.sensor_adr[sensor_id]
        data_dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[sensor_adr:sensor_adr+data_dim]

    def get_rfoot_body_pos(self):
        return self.data.body(self.rfoot_body_name).xpos.copy()

    def get_lfoot_body_pos(self):
        return self.data.body(self.lfoot_body_name).xpos.copy()

    def change_position_to_rotated_world_frame(
        self, position
    ):  # TODO: check if this is correct
        """change position to rotated world frame with yaw rotation the same as root bodt"""
        root_xpos = self.get_root_body_pos()
        root_xmat = self.data.xmat[1].copy().reshape(3, 3)  # xRb
        yaw = np.arctan2(root_xmat[1, 0], root_xmat[0, 0])
        wRb = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        return np.dot(wRb.T, position - root_xpos)

    def change_positions_to_rotated_world_frame(
        self, positions
    ):  # TODO: check if this is correct
        """
        change positions to rotated world frame with yaw rotation the same as root bodt
        Args:
            positions: np.array of shape (3, N)
        """
        root_xpos = self.get_root_body_pos().reshape(3, 1)
        root_xmat = self.data.xmat[1].copy().reshape(3, 3)  # xRb
        yaw = np.arctan2(root_xmat[1, 0], root_xmat[0, 0])
        wRb = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        return np.dot(wRb.T, positions - root_xpos)

    def get_rfoot_keypoint_pos(self):
        """get five foot points in world frame"""
        center_xpos = self.data.geom_xpos[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, self.rfoot_collision_geom_name
            )
        ]
        xmat = self.data.geom_xmat[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, self.rfoot_collision_geom_name
            )
        ].reshape(3, 3)
        point1 = center_xpos + np.dot(xmat, np.array([0.04, 0.1175, -0.0115]))
        point2 = center_xpos + np.dot(xmat, np.array([0.04, -0.1175, -0.0115]))
        point3 = center_xpos + np.dot(xmat, np.array([-0.04, 0.1175, -0.0115]))
        point4 = center_xpos + np.dot(xmat, np.array([-0.04, -0.1175, -0.0115]))
        center_xpos = center_xpos + np.dot(xmat, np.array([0, 0, -0.0115]))
        return [center_xpos, point1, point2, point3, point4]

    def get_lfoot_keypoint_pos(self):
        """get five foot points in world frame"""
        center_xpos = self.data.geom_xpos[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, self.lfoot_collision_geom_name
            )
        ]
        xmat = self.data.geom_xmat[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, self.lfoot_collision_geom_name
            )
        ].reshape(3, 3)
        point1 = center_xpos + np.dot(xmat, np.array([0.04, 0.1175, -0.0115]))
        point2 = center_xpos + np.dot(xmat, np.array([0.04, -0.1175, -0.0115]))
        point3 = center_xpos + np.dot(xmat, np.array([-0.04, 0.1175, -0.0115]))
        point4 = center_xpos + np.dot(xmat, np.array([-0.04, -0.1175, -0.0115]))
        center_xpos = center_xpos + np.dot(xmat, np.array([0, 0, -0.0115]))
        return [center_xpos, point1, point2, point3, point4]

    def get_rfoot_floor_contacts(self):
        """
        Returns list of right foot and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        rcontacts = []
        floor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.floor_body_name
        )
        rfoot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.rfoot_body_name
        )
        for i, c in enumerate(contacts):
            geom1_is_floor = self.model.geom_bodyid[c.geom1] == floor_id
            geom2_is_rfoot = self.model.geom_bodyid[c.geom2] == rfoot_id
            if geom1_is_floor and geom2_is_rfoot:
                rcontacts.append((i, c))
        return rcontacts

    def get_lfoot_floor_contacts(self):
        """
        Returns list of left foot and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        lcontacts = []
        floor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.floor_body_name
        )
        lfoot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.lfoot_body_name
        )
        for i, c in enumerate(contacts):
            geom1_is_floor = self.model.geom_bodyid[c.geom1] == floor_id
            geom2_is_lfoot = self.model.geom_bodyid[c.geom2] == lfoot_id
            if geom1_is_floor and geom2_is_lfoot:
                lcontacts.append((i, c))
        return lcontacts

    def get_rfoot_grf(self):
        """
        Returns total Ground Reaction Force on right foot.
        """
        right_contacts = self.get_rfoot_floor_contacts()
        rfoot_grf = 0
        for i, con in right_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            rfoot_grf += np.linalg.norm(c_array)
        return rfoot_grf

    def get_lfoot_grf(self):
        """
        Returns total Ground Reaction Force on left foot.
        """
        left_contacts = self.get_lfoot_floor_contacts()
        lfoot_grf = 0
        for i, con in left_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            lfoot_grf += np.linalg.norm(c_array)
        return lfoot_grf

    def get_body_vel(self, body_name, frame=0):
        """
        Returns translational and rotational velocity of a body in body-centered frame,
        world/local orientation.
        """
        body_vel = np.zeros(6)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, body_id, body_vel, frame
        )
        return [body_vel[3:6], body_vel[0:3]]

    def get_rfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of right foot.
        """
        rfoot_vel = np.zeros(6)
        rfoot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.rfoot_body_name
        )
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, rfoot_id, rfoot_vel, frame
        )
        return [rfoot_vel[3:6], rfoot_vel[0:3]]

    def get_lfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of left foot.
        """
        lfoot_vel = np.zeros(6)
        lfoot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.lfoot_body_name
        )
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, lfoot_id, lfoot_vel, frame
        )
        return [lfoot_vel[3:6], lfoot_vel[0:3]]

    def get_object_xpos_by_name(self, obj_name, object_type):
        if object_type == "OBJ_BODY":
            return self.data.body(obj_name).xpos
        elif object_type == "OBJ_GEOM":
            return self.data.geom(obj_name).xpos
        elif object_type == "OBJ_SITE":
            return self.data.site(obj_name).xpos
        else:
            raise Exception("object type should either be OBJ_BODY/OBJ_GEOM/OBJ_SITE.")

    def get_object_xquat_by_name(self, obj_name, object_type):
        if object_type == "OBJ_BODY":
            return self.data.body(obj_name).xquat
        if object_type == "OBJ_SITE":
            xmat = self.data.site(obj_name).xmat
            return tf3.quaternions.mat2quat(xmat)
        else:
            raise Exception("object type should be OBJ_BODY/OBJ_SITE.")

    def get_robot_com(self):
        """
        Returns the center of mass of subtree originating at root body
        i.e. the CoM of the entire robot body in world coordinates.
        """
        sensor_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(self.model.nsensor)
        ]
        if "subtreecom" not in sensor_names:
            raise Exception("subtree_com sensor not attached.")
        return self.data.subtree_com[1].copy()

    def get_robot_linmom(self):
        """
        Returns linear momentum of robot in world coordinates.
        """
        sensor_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(self.model.nsensor)
        ]
        if "subtreelinvel" not in sensor_names:
            raise Exception("subtree_linvel sensor not attached.")
        linvel = self.data.subtree_linvel[1].copy()
        total_mass = self.get_robot_mass()
        return linvel * total_mass

    def get_robot_angmom(self):
        """
        Return angular momentum of robot's CoM about the world origin.
        """
        sensor_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(self.model.nsensor)
        ]
        if "subtreeangmom" not in sensor_names:
            raise Exception("subtree_angmom sensor not attached.")
        com_pos = self.get_robot_com()
        lin_mom = self.get_robot_linmom()
        return self.data.subtree_angmom[1] + np.cross(com_pos, lin_mom)

    def check_rfoot_floor_collision(self):
        """
        Returns True if there is a collision between right foot and floor.
        """
        return len(self.get_rfoot_floor_contacts()) > 0

    def check_lfoot_floor_collision(self):
        """
        Returns True if there is a collision between left foot and floor.
        """
        return len(self.get_lfoot_floor_contacts()) > 0

    def check_bad_collisions(self):
        """
        Returns True if there are collisions other than feet-floor.
        """
        num_rcons = len(self.get_rfoot_floor_contacts())
        num_lcons = len(self.get_lfoot_floor_contacts())
        return (num_rcons + num_lcons) != self.data.ncon

    def check_self_collisions(self):
        """
        Returns True if there are collisions other than any-geom-floor.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        floor_contacts = []
        floor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.floor_body_name
        )
        for i, c in enumerate(contacts):
            geom1_is_floor = self.model.geom_bodyid[c.geom1] == floor_id
            geom2_is_floor = self.model.geom_bodyid[c.geom2] == floor_id
            if geom1_is_floor or geom2_is_floor:
                floor_contacts.append((i, c))
        return len(floor_contacts) != self.data.ncon

    def check_body_lean(self):
        rot_mat = self.data.xmat[1].copy()
        return np.arccos(rot_mat[8]) * 180 / np.pi > 20

    def get_projected_gravity_vec(self):
        # xmat has global orientation of the object.
        # https://github.com/google-deepmind/dm_control/issues/160
        rot_mat = self.data.xmat[
            1
        ]  # wXb -> index 0,3,6 ; wYb -> index 1,4,7 ; wZb -> index 2,5,8
        return rot_mat[6:9].copy()

    def get_pd_target(self):
        return [self.current_pos_target, self.current_vel_target]

    def set_pd_gains(self, kp, kv):
        assert kp.size == self.model.nu
        assert kv.size == self.model.nu
        self.kp = kp.copy()
        self.kv = kv.copy()
        return

    def step_pd(self, p, v):
        self.current_pos_target = p.copy()
        self.current_vel_target = v.copy()
        target_angles = self.current_pos_target
        target_speeds = self.current_vel_target

        assert type(target_angles) is np.ndarray
        assert type(target_speeds) is np.ndarray

        curr_angles = self.get_act_joint_positions()
        curr_speeds = self.get_act_joint_velocities()

        perror = target_angles - curr_angles
        verror = target_speeds - curr_speeds

        assert self.kp.size == perror.size
        assert self.kv.size == verror.size
        assert perror.size == verror.size
        return self.kp * perror + self.kv * verror

    def step_d(self, v):
        self.current_vel_target = v.copy()
        target_speeds = self.current_vel_target

        assert type(target_speeds) is np.ndarray

        curr_speeds = self.get_act_joint_velocities()

        verror = target_speeds - curr_speeds

        assert self.kv.size == verror.size
        return self.kv * verror

    def set_motor_torque(self, torque):
        """
        Apply torques to motors.
        """
        if isinstance(torque, np.ndarray):
            assert torque.shape == (self.nu(),)
            ctrl = torque.tolist()
        elif isinstance(torque, list):
            assert len(torque) == self.nu()
            ctrl = np.copy(torque)
        else:
            raise Exception("motor torque should be list of ndarray.")
        try:
            self.data.ctrl[:] = ctrl
        except Exception as e:
            print("Could not apply motor torque.")
            print(e)
        return

    def step(self):
        """
        Increment simulation by one step.
        """
        mujoco.mj_step(self.model, self.data)
