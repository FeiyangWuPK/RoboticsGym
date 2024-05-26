# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from roboticsgym.envs.cfg.base.base_config import BaseConfig, ConfigObj


def standing_joint_pos_tracking(CurrJoint_pos, RefJoint_pos):
    joint_pos_error = np.sum((CurrJoint_pos - RefJoint_pos) ** 2)

    return np.exp(-5 * joint_pos_error)


def standing_joint_vel_tracking(CurrJoint_vel, RefJoint_vel):
    joint_vel_error = np.sum((CurrJoint_vel - RefJoint_vel) ** 2)
    return np.exp(-0.1 * joint_vel_error)


def standing_root_pos_tracking(Currbase_pos, Refbase_pos):
    position_error = np.sum((Currbase_pos[:3] - Refbase_pos[:3]) ** 2)
    ref_quat = np.array(
        [Refbase_pos[3], Refbase_pos[4], Refbase_pos[5], Refbase_pos[6]]
    )
    current_quat = np.array(
        [Currbase_pos[4], Currbase_pos[5], Currbase_pos[6], Currbase_pos[3]]
    )
    rotation_error = np.sum((current_quat - ref_quat) ** 2)
    return np.exp(-20 * position_error - 10 * rotation_error)


def standing_root_vel_tracking(Currbase_vel, Refbase_vel):
    linear_vel_error = np.sum((Currbase_vel[:3] - Refbase_vel[:3]) ** 2)
    angular_vel_error = np.sum((Currbase_vel[3:] - Refbase_vel[3:]) ** 2)
    return np.exp(-2 * linear_vel_error - 0.2 * angular_vel_error)
    # return np.exp(-2 * linear_vel_error)


def standing_endeffector_tracking(end_effector_pos, ref_end_effector_pos):
    end_effector_pos = end_effector_pos.reshape((12,))
    ref_end_effector_pos = ref_end_effector_pos.reshape((12,))
    squared_diff = np.sum(np.square(end_effector_pos - ref_end_effector_pos) ** 2)
    reward = np.exp(-40 * squared_diff)
    return reward


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
    rfoot_penalty = (rfoot_grf < 1) * np.sum(np.square(action[14:16]))
    return lfoot_penalty + rfoot_penalty


class DigitEnvConfig(BaseConfig):
    seed = 0

    class env(ConfigObj):
        max_time = 20.0  # (s)
        sim_dt = 0.0005  # (s)
        obs_dim = 136
        value_obs_dim = 136
        hist_len_s = 0.1  # (s)
        hist_interval_s = 0.03  # (s).
        act_dim = 20  # target joint pos

    class terrain(ConfigObj):
        terrain_type = "flat"  # flat, test_rough, rough_random???
        terrain_path = "terrain_info"

    class reset_state(ConfigObj):
        random_dof_reset = False
        p_std = 0.03
        v_std = 0.06
        root_p_uniform = 0.4
        root_v_std = 0.1
        random_dof_names = [
            "left-hip-roll",
            "left-hip-yaw",
            "left-hip-pitch",
            "left-knee",
            "left-toe-A",
            "left-toe-B",
            "right-hip-roll",
            "right-hip-yaw",
            "right-hip-pitch",
            "right-knee",
            "right-toe-A",
            "right-toe-B",
            "left-tarsus",
            "left-toe-pitch",
            "left-toe-roll",
            "right-tarsus",
            "right-toe-pitch",
            "right-toe-roll",
        ]

    class commands(ConfigObj):
        curriculum = False  # when true, vel range should be changed
        max_curriculum = 1.0
        resampling_time = 10.0  # time before command are changed[s]

        class ranges(ConfigObj):
            # x_vel_range = [0.,1.5] # min max [m/s]
            # y_vel_range = [-0.4,0.4] # min max [m/s]
            # ang_vel_range = [-0.4,0.4] # min max [rad/s]
            x_vel_range = [0.0, 1.4]  # min max [m/s]
            y_vel_range = [-0.2, 0.2]  # min max [m/s]
            ang_vel_range = [-0.3, 0.3]  # min max [rad/s]

            cut_off = 0.1

    class control(ConfigObj):
        mbc_control = False  # if true, mbc action is used in def step()
        control_type = "PD"  # PD: PD control, T: torques
        action_scale = 0.1
        control_dt = 0.005
        # control_dt = 0.03
        lower_motor_index = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        # default_kp = np.array([1400, 1000, 1167, 1300, 533, 533, 500, 500, 500, 500, 1400, 1000, 1167, 1300, 533, 533, 500, 500, 500, 500])
        # default_kd = np.array([5,5,5,5,5,5, 5,5,5,5, 5,5,5,5,5,5, 5,5,5,5])
        # default_kp = np.array([460, 430, 600, 625, 150, 150, 100, 100, 100, 100,  460, 430, 600, 625, 150, 150, 100, 100, 100, 100])
        # default_kd = np.array([8.0, 8.0, 8.0, 5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 8.0, 8.0, 8.0, 5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

    class vis_record(ConfigObj):
        visualize = False  # should set to false when training
        record = True  # should visualize true
        record_fps = 15
        record_env = 0  # -1 if you don't wanna recording. recording should always be done in env 0
        snapshot_gap = 10

    class domain_randomization(ConfigObj):
        is_true = True
        action_delay = 0.002  # TODO: isn't it to large?
        # friction_noise = [0.4, 2.0] # scaling
        kp_noise = [0.9, 1.1]
        kd_noise = [0.9, 1.1]
        joint_friction = [0.0, 0.7]

    class rewards(ConfigObj):
        class scales(ConfigObj):

            # joint_pos_tracking = 0.4
            # joint_vel_tracking = 0.
            # root_pos_tracking = 0.2
            # root_vel_tracking = 0.1
            # endeffector_tracking = 0.3

            # # walking-forward
            # joint_pos_tracking = 0.4
            # joint_vel_tracking = 0.05
            # root_pos_tracking = 0.15
            # root_vel_tracking = 0.1
            # endeffector_tracking = 0.3
            # swing_foot_fix_penalty = -0.

            # jumping
            joint_pos_tracking = 0.4
            joint_vel_tracking = 0.05
            root_pos_tracking = 0.2
            root_vel_tracking = 0.1
            endeffector_tracking = 0.25
            swing_foot_fix_penalty = -0.0

            # reward_reaching
            # joint_pos_tracking = 0.5
            # joint_vel_tracking = 0.05
            # root_pos_tracking = 0.15
            # root_vel_tracking = 0.1
            # endeffector_tracking = 0.2
            # swing_foot_fix_penalty = -1

            # lin_vel_tracking = 2.
            # ang_vel_tracking = 1.5
            # z_vel_penalty = -0.01
            # roll_pitch_penalty = -0.1
            # torque_penalty = -2e-5
            # base_orientation_penalty = -0.1
            # foot_lateral_distance_penalty = -0.
            # swing_foot_fix_penalty = -0.1
            # termination = 0.

        @classmethod
        def update_scales(cls, task_name):
            task_config = DigitEnvConfig.TASK_CONFIG.get(task_name)
            if not task_config:
                raise ValueError(f"Unknown task: {task_name}")

            reward_weights = task_config.get("reward_weights", {})
            for reward_name, weight in reward_weights.items():
                setattr(cls.scales, reward_name, weight)

    class normalization(ConfigObj):
        class obs_scales(ConfigObj):
            lin_vel = 2.0
            ang_vel = 2.0
            dof_pos = 1.0
            dof_vel = 0.05

        clip_obs = (
            100.0  # NOTE: clipped action wihtout scaling is included in observation!
        )
        clip_act = 100.0  # NOTE: make sure to change these when torque control

    class obs_noise(ConfigObj):
        is_true = True
        lin_vel_std = 0.15
        ang_vel_std = 0.15
        dof_pos_std = 0.175
        dof_vel_std = 0.15
        projected_gravity_std = 0.075

    TASK_CONFIG = {
        "standing": {
            "reward_weights": {
                "joint_pos_tracking": 0.4,
                "joint_vel_tracking": 0.05,
                "root_pos_tracking": 0.15,
                "root_vel_tracking": 0.1,
                "endeffector_tracking": 0.3,
                "swing_foot_fix_penalty": -0.0,
            },
            "kp": np.array(
                [
                    100,
                    100,
                    88,
                    96,
                    50,
                    50,
                    50,
                    50,
                    50,
                    50,
                    100,
                    100,
                    88,
                    96,
                    50,
                    50,
                    50,
                    50,
                    50,
                    50,
                ]
            ),
            "kd": np.array(
                [
                    10.0,
                    10.0,
                    8.0,
                    9.6,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    10.0,
                    10.0,
                    8.0,
                    9.6,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                ]
            ),
            "indices_to_keep_qpos": [
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
                14,
                15,
                16,
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
                41,
                42,
                43,
                45,
                50,
                55,
                56,
                57,
                58,
                59,
                60,
            ],
            "indices_to_keep_qvel": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                12,
                13,
                14,
                16,
                20,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                36,
                37,
                38,
                40,
                44,
                48,
                49,
                50,
                51,
                52,
                53,
            ],
            "reward_functions": {
                "joint_pos_tracking": {
                    "func": standing_joint_pos_tracking,
                    "args": ["qpos_filt[7:]", "ref_qpos_filt[7:]"],
                },
                "joint_vel_tracking": {
                    "func": standing_joint_vel_tracking,
                    "args": ["qvel_filt[6:]", "ref_qvel_filt[6:]"],
                },
                "root_pos_tracking": {
                    "func": standing_root_pos_tracking,
                    "args": ["qpos_filt[:7]", "ref_qpos_filt[:7]"],
                },
                "root_vel_tracking": {
                    "func": standing_root_vel_tracking,
                    "args": ["qvel_filt[:6]", "ref_qvel_filt[:6]"],
                },
                "endeffector_tracking": {
                    "func": standing_endeffector_tracking,
                    "args": ["endeffector_filt", "ref_ee_pos"],
                },
                "swing_foot_fix_penalty": {
                    "func": swing_foot_fix_penalty,
                    "args": [
                        "self._interface.get_lfoot_grf()",
                        "self._interface.get_rfoot_grf()",
                        "self.action",
                    ],
                },
            },
            "termination_conditions": [
                "root_vel_crazy_check",
                "self_collision_check",
                "body_lean_check",
                "ref_traj_step_check",
            ],
        },
        "walking_forward": {
            "reward_weights": {
                "joint_pos_tracking": 0.4,
                "joint_vel_tracking": 0.05,
                "root_pos_tracking": 0.15,
                "root_vel_tracking": 0.1,
                "endeffector_tracking": 0.3,
                "swing_foot_fix_penalty": -0.0,
            },
            "kp": np.array(
                [
                    800,
                    600,
                    800,
                    1000,
                    200,
                    200,
                    100,
                    100,
                    100,
                    100,
                    800,
                    600,
                    800,
                    1000,
                    200,
                    200,
                    100,
                    100,
                    100,
                    100,
                ]
            ),
            "kd": np.array(
                [
                    8.0,
                    8.0,
                    8.0,
                    5,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    8.0,
                    8.0,
                    8.0,
                    5,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                ]
            ),
            "indices_to_keep_qpos": [
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
                14,
                15,
                16,
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
                41,
                42,
                43,
                45,
                50,
                55,
                56,
                57,
                58,
                59,
                60,
            ],
            "indices_to_keep_qvel": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                12,
                13,
                14,
                16,
                20,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                36,
                37,
                38,
                40,
                44,
                48,
                49,
                50,
                51,
                52,
                53,
            ],
            "reward_functions": {
                "joint_pos_tracking": {
                    "func": standing_joint_pos_tracking,
                    "args": ["qpos_filt[7:]", "ref_qpos_filt[7:]"],
                },
                "joint_vel_tracking": {
                    "func": standing_joint_vel_tracking,
                    "args": ["qvel_filt[6:]", "ref_qvel_filt[6:]"],
                },
                "root_pos_tracking": {
                    "func": standing_root_pos_tracking,
                    "args": ["qpos_filt[:7]", "ref_qpos_filt[:7]"],
                },
                "root_vel_tracking": {
                    "func": standing_root_vel_tracking,
                    "args": ["qvel_filt[:6]", "ref_qvel_filt[:6]"],
                },
                "endeffector_tracking": {
                    "func": standing_endeffector_tracking,
                    "args": ["endeffector_filt", "ref_ee_pos"],
                },
                "swing_foot_fix_penalty": {
                    "func": swing_foot_fix_penalty,
                    "args": [
                        "self._interface.get_lfoot_grf()",
                        "self._interface.get_rfoot_grf()",
                        "self.action",
                    ],
                },
            },
            "termination_conditions": [
                "root_vel_crazy_check",
                "self_collision_check",
                "body_lean_check",
                "ref_traj_step_check",
            ],
        },
    }

    # ***********qpos: 61********************

    # [0-6] base, x y z qw qx qy qz
    # [7] left-hip-roll
    # [8] left-hip-yaw
    # [9] left-hip-pitch
    # [10-13] left-achillies-rod
    # [14] left-knee
    # [15] left-shin
    # [16] left-tarsus
    # [17] left-heel-spring
    # [18] left-toe-A
    # [19-22] left-toe-A-rod
    # [23] left-toe-B
    # [24-27] left-toe-B-rod
    # [28] left-toe-pitch
    # [29] left-toe-roll
    # [30] left-shoulder-roll
    # [31] left-shoulder-pitch
    # [32] left-shoulder-yaw
    # [33] left-elbow
    # [34] right-hip-roll
    # [35] right-hip-yaw
    # [36] right-hip-pitch
    # [37-40] right-achillies-rod
    # [41] right-knee
    # [42] right-shin
    # [43] right-tarsus
    # [44] right-heel-spring
    # [45] right-toe-A
    # [46-49] right-toe-A-rod
    # [50] right-toe-B
    # [51-54] right-toe-B-rod
    # [55] right-toe-pitch
    # [56] right-toe-roll
    # [57] right-shoulder-roll
    # [58] right-shoulder-pitch
    # [59] right-shoulder-yaw
    # [60] right-elbow

    # qvel 54: env.model.jnt_dofadr, env.init_qvel.size

    # [0-5] base
    # [6] left-hip-roll
    # [7] left-hip-yaw
    # [8] left-hip-pitch
    # [9-11] left-achillies-rod
    # [12] left-knee
    # [13] left-shin
    # [14] left-tarsus
    # [15] left-heel-spring
    # [16] left-toe-A
    # [17-19] left-toe-A-rod
    # [20] left-toe-B
    # [21-23] left-toe-B-rod
    # [24] left-toe-pitch
    # [25] left-toe-roll
    # [26] left-shoulder-roll
    # [27] left-shoulder-pitch
    # [28] left-shoulder-yaw
    # [29] left-elbow
    # [30] right-hip-roll
    # [31] right-hip-yaw
    # [32] right-hip-pitch
    # [33-35] right-achillies-rod
    # [36] right-knee
    # [37] right-shin
    # [38] right-tarsus
    # [39] right-heel-spring
    # [40] right-toe-A
    # [41-43] right-toe-A-rod
    # [44] right-toe-B
    # [45-47] right-toe-B-rod
    # [48] right-toe-pitch
    # [49] right-toe-roll
    # [50] right-shoulder-roll
    # [51] right-shoulder-pitch
    # [52] right-shoulder-yaw
    # [53] right-elbow

    # motor/actuators: 20
    # joint_names = ["left-hip-roll",
    #                 "left-hip-yaw",
    #                 "left-hip-pitch",
    #                 "left-knee",
    #                 "left-toe-A",
    #                 "left-toe-B",
    #                 "left-shoulder-roll",
    #                 "left-shoulder-pitch",
    #                 "left-shoulder-yaw",
    #                 "left-elbow",
    #                 "right-hip-roll",
    #                 "right-hip-yaw",
    #                 "right-hip-pitch",
    #                 "right-knee",
    #                 "right-toe-A",
    #                 "right-toe-B",
    #                 "right-shoulder-roll",
    #                 "right-shoulder-pitch",
    #                 "right-shoulder-yaw",
    #                 "right-elbow"
    #                 ]
