import math
import pickle
from arm_cassie_env.cassie_env.quaternion_function import *
from arm_cassie_env.cassie_env.cassieRLEnvMultiTraj import *

class cassieRLEnvMirror(cassieRLEnvMultiTraj):
    def __init__(self, visual=False, acbuf=False, record_for_reward_inference = False):
        super().__init__(visual)
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

        self.action_buf=[]
        self.acbuf=acbuf
        self.height_rec=[]
        self.noisy=False
        self.delay=False
        self.whole_state_buffer = []
        self.useful_recorded_data = []
        self.record_for_reward_inference = record_for_reward_inference
    
    def get_state(self):
        # if len(self.state_buffer) > 0:
        # 	random_index = random.randint(0, len(self.state_buffer)-1)
        # 	state = self.state_buffer[random_index]
        # else:
        # 	state = self.cassie_state

        state = self.cassie_state
        rp ,rv = self.get_kin_next_state()
        ref_pos= np.copy(rp)
        ref_vel=np.copy(rv) 
        if self.phase < 14:
            pos_index = np.array([2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
            vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
            quaternion = euler2quat(z=self.orientation, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)
            new_orientation = quaternion_product(iquaternion, state.pelvis.orientation[:])
            #print(new_orientation)
            new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
            #print(new_translationalVelocity)
            new_translationalAcceleration = rotate_by_quaternion(state.pelvis.translationalAcceleration[:], iquaternion)
            new_rotationalVelocity = rotate_by_quaternion(state.pelvis.rotationalVelocity[:], quaternion)
            useful_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], state.motor.position[:], new_translationalVelocity[:], state.pelvis.rotationalVelocity[:], state.motor.velocity[:], new_translationalAcceleration[:], state.joint.position[:], state.joint.velocity[:]]))
            return np.concatenate([useful_state, ref_pos[pos_index], ref_vel[vel_index]])
        else:
            pos_index = np.array([2,3,4,5,6,21,22,23,28,29,30,34,7,8,9,14,15,16,20])
            vel_index = np.array([0,1,2,3,4,5,19,20,21,25,26,27,31,6,7,8,12,13,14,18])
            ref_vel[1] = -ref_vel[1]
            euler = quaternion2euler(ref_pos[3:7])
            euler[0] = -euler[0]
            euler[2] = -euler[2]
            ref_pos[3:7] = euler2quat(z=euler[2],y=euler[1],x=euler[0])
            quaternion = euler2quat(z=-self.orientation, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)

            pelvis_euler = quaternion2euler(np.copy(state.pelvis.orientation[:]))
            pelvis_euler[0] = -pelvis_euler[0]
            pelvis_euler[2] = -pelvis_euler[2]
            pelvis_orientation = euler2quat(z=pelvis_euler[2],y=pelvis_euler[1],x=pelvis_euler[0])

            translational_velocity = np.copy(state.pelvis.translationalVelocity[:])
            translational_velocity[1] = -translational_velocity[1]

            translational_acceleration = np.copy(state.pelvis.translationalAcceleration[:])
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
            new_translationalVelocity = rotate_by_quaternion(translational_velocity, iquaternion)
            new_translationalAcceleration = rotate_by_quaternion(translational_acceleration, iquaternion)
            new_rotationalVelocity = rotate_by_quaternion(rotational_velocity, quaternion)

            useful_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], motor_position, new_translationalVelocity[:], rotational_velocity, motor_velocity, new_translationalAcceleration[:], joint_position, joint_velocity]))
            
            return np.concatenate([useful_state, ref_pos[pos_index], ref_vel[vel_index]])
    def step_simulation(self, action):
        # qpos = np.copy(self.sim.qpos())
        # qvel = np.copy(self.sim.qvel())

        pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        # vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        ref_pos, ref_vel = self.get_kin_next_state()
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
        self.current_action = np.array(action)
        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.torque[i] = 0 # Feedforward torque
            self.u.leftLeg.motorPd.pTarget[i] = target[i]
            self.u.leftLeg.motorPd.pGain[i] = self.P[i]
            self.u.leftLeg.motorPd.dTarget[i] = 0
            self.u.leftLeg.motorPd.dGain[i] = self.D[i]
            self.u.rightLeg.motorPd.torque[i] = 0 # Feedforward torque
            self.u.rightLeg.motorPd.pTarget[i] = target[i+5]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i+5]
            self.u.rightLeg.motorPd.dTarget[i] = 0
            self.u.rightLeg.motorPd.dGain[i] = self.D[i+5]

        self.state_buffer.append(self.sim.step_pd(self.u))
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
        
        self.cassie_state = self.state_buffer[len(self.state_buffer) - 1]

    def step(self, action):
        for _ in range(self.control_rate):
            self.step_simulation(action)

        self.whole_state_buffer.append(self.state_buffer[-1])
        left_foot_force, right_foot_force = self.sim.get_foot_forces()
        center_of_mass_pos = self.sim.center_of_mass_position()
        center_of_mass_vel = self.sim.center_of_mass_velocity() # list of 3
        center_of_mass_angular_momentum = self.sim.angular_momentum() # list of 3
        center_of_mass_centroid_inertia = self.sim.centroid_inertia() # 3 by 3 matrix, list of 9
        feet_pos = self.sim.foot_pos()
        self.useful_recorded_data.append([left_foot_force, right_foot_force] + center_of_mass_pos + feet_pos + center_of_mass_vel+ center_of_mass_angular_momentum + center_of_mass_centroid_inertia)

        height = self.sim.qpos()[2]
        self.time += 1
        self.phase += 1

        if self.phase >= self.max_phase:
            self.phase = 0
            self.counter +=1
        #print("height", height)

        done = not(height > 0.4 and height < 100.0) or self.time >= self.time_limit
        yaw = quat2yaw(self.sim.qpos()[3:7])
        if self.visual:
            self.render()

        reward = self.compute_reward()
        #print(reward)
        if reward < 0.3:
            done = True

        if done and self.record_for_reward_inference:
            # save the whole state_buffer for reward inference
            import time
            t = time.strftime("%Y_%m_%d_%H_%M_%S")
            os.makedirs("logs/reward_inference/traj_state_buffer", exist_ok=True)
            with open(f"logs/reward_inference/traj_state_buffer/buffer_{t}", "wb") as fp:
                pickle.dump(self.whole_state_buffer, fp)
            self.whole_state_buffer = []

            os.makedirs("logs/reward_inference/traj_useful_recorded_data", exist_ok=True)
            with open(f"logs/reward_inference/traj_useful_recorded_data/buffer_{t}", "wb") as fp2:
                pickle.dump(self.useful_recorded_data, fp2)
            self.useful_recorded_data = []

        return self.get_state(), reward, done, {}

    def reset(self):
        if self.time != 0 :
            self.rew_ref_buf = self.rew_ref / self.time
            self.rew_spring_buf = self.rew_spring / self.time
            self.rew_ori_buf = self.rew_ori / self.time
            self.rew_vel_buf = self.rew_vel / self.time
            self.reward_buf = self.reward # / self.time
            self.time_buf = self.time
        
        self.rew_ref = 0
        self.rew_spring = 0
        self.rew_ori = 0
        self.rew_vel = 0
        self.rew_termin = 0
        self.rew_cur = 0
        self.reward = 0
        self.omega = 0
        self.height_rec=[]

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
        self.sim.c = cassie_sim_init(self.model.encode('utf-8'), False)
        qpos, qvel = self.get_kin_state()
        qpos[3:7] = quaternion
        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)
        self.cassie_state = self.sim.step_pd(self.u)		

        return self.get_state()

    def reset_by_speed(self, speed, y_speed=0, phase=None):
        self.orientation = 0
        self.speed = speed#(random.randint(-10, 10)) / 10.0
        self.y_speed = 0
        orientation = self.orientation + (random.randint(0, 1) * 2 - 1) * np.pi / 10
        quaternion = euler2quat(z=orientation, y=0, x=0)
        if phase is None:
            self.phase = random.randint(0, 27)
        else:
            self.phase = phase
        self.time = 0
        self.counter = 0
        cassie_sim_free(self.sim.c)
        self.sim.c = cassie_sim_init()
        qpos, qvel = self.get_kin_state()
        qpos[3:7] = quaternion
        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)
        self.cassie_state = self.sim.step_pd(self.u)
        return self.get_state()

    def reset_by_phase(self, phase):
        self.orientation = 0
        self.speed = (random.randint(-10, 10)) / 10
        orientation = 0#self.orientation + random.randint(-20, 20) * np.pi / 100
        quaternion = euler2quat(z=orientation, y=0, x=0)
        self.phase = phase
        self.time = 0
        self.counter = 0
        cassie_sim_free(self.sim.c)
        self.sim.c = cassie_sim_init()
        qpos, qvel = self.get_kin_state()
        qpos[3:7] = quaternion
        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)
        self.cassie_state = self.sim.step_pd(self.u)
        return self.get_state()
    
    def compute_reward(self):
        ref_pos, ref_vel = self.get_kin_state()
        ref_pos = np.array(ref_pos)
        ref_vel = np.array(ref_vel)
        weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
        joint_penalty = 0
        joint_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        for i in range(10):
            error = weight[i] * (ref_pos[joint_index[i]]-self.sim.qpos()[joint_index[i]])**2
            joint_penalty += error*30

        pelvis_pos = np.copy(self.cassie_state.pelvis.position[:])
        com_penalty = (pelvis_pos[0]-ref_pos[0])**2 + (pelvis_pos[1]-ref_pos[1])**2 + (self.sim.qvel()[2])**2

        # yaw = quat2yaw(self.sim.qpos()[3:7])

        # orientation_penalty = (self.sim.qpos()[4])**2+(self.sim.qpos()[5])**2+(yaw - self.orientation)**2
        orientation_penalty = np.linalg.norm(self.sim.qpos()[3:7] - ref_pos[3:7]) ** 2

        overall_pos_penalty = np.linalg.norm(self.sim.qpos() - ref_pos) ** 2

        spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
        spring_penalty *= 1000

        # speed_penalty = (self.sim.qvel()[0] - ref_vel[0])**2 + (self.sim.qvel()[1] - ref_vel[1])**2
        cur_qvel = np.array(self.sim.qvel())
        speed_penalty = np.linalg.norm(cur_qvel[vel_index] - ref_vel[vel_index]) ** 2
        total_reward = 0.3 * np.exp(-joint_penalty) + 0.3 * np.exp(-com_penalty) + 0.2 * np.exp(-10*orientation_penalty) + 0.1 * np.exp(-overall_pos_penalty) + 0.1 * np.exp(-speed_penalty)

        forward_reward = 0.25 * cur_qvel[0]

        control_cost = 0.25 * np.linalg.norm(self.current_action) ** 2

        total_reward += forward_reward - control_cost

        self.rew_ref += 0.5*np.exp(-joint_penalty)
        self.rew_spring += 0.1*np.exp(-spring_penalty)
        self.rew_ori += 0.1*np.exp(-orientation_penalty)
        self.rew_vel += 0.3*np.exp(-com_penalty)
        self.reward += total_reward		

        return total_reward