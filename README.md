
# Cassie

The Cassie joint definition from cassie.xml.

[ 0] Pelvis x
[ 1] Pelvis y
[ 2] Pelvis z
[ 3] Pelvis orientation qw
[ 4] Pelvis orientation qx
[ 5] Pelvis orientation qy
[ 6] Pelvis orientation qz
[ 7] Left hip roll         (Motor [0])
[ 8] Left hip yaw          (Motor [1])
[ 9] Left hip pitch        (Motor [2])
[10] Left achilles rod qw
[11] Left achilles rod qx
[12] Left achilles rod qy
[13] Left achilles rod qz
[14] Left knee             (Motor [3])
[15] Left shin                        (Joint [0])
[16] Left tarsus                      (Joint [1])
[17] Left heel spring
[18] Left foot crank
[19] Left plantar rod
[20] Left foot             (Motor [4], Joint [2])
[21] Right hip roll        (Motor [5])
[22] Right hip yaw         (Motor [6])
[23] Right hip pitch       (Motor [7])
[24] Right achilles rod qw
[25] Right achilles rod qx
[26] Right achilles rod qy
[27] Right achilles rod qz
[28] Right knee            (Motor [8])
[29] Right shin                       (Joint [3])
[30] Right tarsus                     (Joint [4])
[31] Right heel spring
[32] Right foot crank
[33] Right plantar rod
[34] Right foot            (Motor [9], Joint [5])

motor/actuators: 10

# Digit

The Digit joint definition from digit.xml.
mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, joint_id)
env.model.jnt_bodyid

qpos 61: env.model.jnt_qposadr, env.init_qpos.size

[0-6] base, x y z qw qx qy qz
[7] left-hip-roll
[8] left-hip-yaw
[9] left-hip-pitch
[10-13] left-achillies-rod
[14] left-knee
[15] left-shin
[16] left-tarsus
[17] left-heel-spring
[18] left-toe-A
[19-22] left-toe-A-rod
[23] left-toe-B
[24-27] left-toe-B-rod
[28] left-toe-pitch
[29] left-toe-roll
[30] left-shoulder-roll
[31] left-shoulder-pitch
[32] left-shoulder-yaw
[33] left-elbow
[34] right-hip-roll
[35] right-hip-yaw
[36] right-hip-pitch
[37-40] right-achillies-rod
[41] right-knee
[42] right-shin
[43] right-tarsus
[44] right-heel-spring
[45] right-toe-A
[46-49] right-toe-A-rod
[50] right-toe-B
[51-54] right-toe-B-rod
[55] right-toe-pitch
[56] right-toe-roll
[57] right-shoulder-roll
[58] right-shoulder-pitch
[59] right-shoulder-yaw
[60] right-elbow

qvel 54: env.model.jnt_dofadr, env.init_qvel.size

[0-5] base
[6] left-hip-roll
[7] left-hip-yaw
[8] left-hip-pitch
[9-11] left-achillies-rod
[12] left-knee
[13] left-shin
[14] left-tarsus
[15] left-heel-spring
[16] left-toe-A
[17-19] left-toe-A-rod
[20] left-toe-B
[21-23] left-toe-B-rod
[24] left-toe-pitch
[25] left-toe-roll
[26] left-shoulder-roll
[27] left-shoulder-pitch
[28] left-shoulder-yaw
[29] left-elbow
[30] right-hip-roll
[31] right-hip-yaw
[32] right-hip-pitch
[33-35] right-achillies-rod
[36] right-knee
[37] right-shin
[38] right-tarsus
[39] right-heel-spring
[40] right-toe-A
[41-43] right-toe-A-rod
[44] right-toe-B
[45-47] right-toe-B-rod
[48] right-toe-pitch
[49] right-toe-roll
[50] right-shoulder-roll
[51] right-shoulder-pitch
[52] right-shoulder-yaw
[53] right-elbow

motor/actuators: 20