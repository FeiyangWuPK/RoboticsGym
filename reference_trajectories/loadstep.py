import numpy as np
import random
import pandas as pd
import math

class CassieTrajectory:
    def __init__(self, filepath):
        self.read_mat(filepath)

    def state(self, t):
        i = int(t % self.num_data)
        return (self.qpos[i], self.qvel[i])

    def action(self, t):
        i = int(t % self.num_data)
        return (self.mpos[i], self.mvel[i], self.torque[i])

    def sample(self):
        i = random.randrange(self.num_data)
        return (self.time[i], self.qpos[i], self.qvel[i])
    
    def read_mat(self, filepath):
        import scipy
        Data = scipy.io.loadmat(filepath)
        Data = Data['cassie_walking_data'][0,0]
        q = Data[0].transpose()
        dq = Data[1].transpose()
        u = Data[2].transpose()
        self.num_data = q.shape[0]
        
        self.qpos = np.zeros((self.num_data, 35))
        self.qpos[:, 0:3] = q[:, 0:3]
        self.qpos[:, 3:7] = np.tile(np.array([1, 0, 0, 0]), (self.num_data,1))
        self.qpos[:, 7:10] = q[:, 6:9]
        self.qpos[:, 14:17] = q[:, 9:12]
        self.qpos[:, 20] = q[:, 12]
        self.qpos[:, 21:24] = q[:, 13:16]
        self.qpos[:, 28:31] = q[:, 16:19]
        self.qpos[:, 34] = q[:, 19]
        
        self.qvel = np.zeros((self.num_data, 32))
        self.qvel[:, 0:3] = dq[:, 0:3]
        self.qvel[:, 3:6] = dq[:, 3:6]
        self.qvel[:, 6:9] = dq[:, 6:9]
        self.qvel[:, 12:15] = dq[:, 9:12]
        self.qvel[:, 18] = dq[:, 12]
        self.qvel[:, 19:22] = dq[:, 13:16]
        self.qvel[:, 25:28] = dq[:, 16:19]
        self.qvel[:, 21] = dq[:, 19]
        
        self.torque = u
        
        self.mpos = np.zeros((self.num_data, 10))
        self.mpos[:, 0:3] = self.qpos[:, 7:10]
        self.mpos[:, 3] = self.qpos[:, 14]
        self.mpos[:, 4] = self.qpos[:, 20]
        self.mpos[:, 5:8] = self.qpos[:, 21:24]
        self.mpos[:, 8] = self.qpos[:, 28]
        self.mpos[:, 9] = self.qpos[:, 34]
        
        self.mvel = np.zeros((self.num_data, 10))
        self.mvel[:, 0:3] = self.qvel[:, 6:9]
        self.mvel[:, 3] = self.qvel[:, 12]
        self.mvel[:, 4:8] = self.qvel[:, 18:22]
        self.mvel[:, 8] = self.qvel[:, 25]
        self.mvel[:, 9] = self.qvel[:, 31]

    def read_csv(self, filepath):
        df = pd.read_csv(filepath, header=None) 
        # df = df.drop(df.columns[0], axis=1)
        self.num_data = len(df.columns)
        print("num_data: ", self.num_data)
        # each column is [time, base_quat_wxyz(4), base_xyz(3), cassie_state(14), arm_state(6)]
        data = df.to_numpy().transpose()

        target_data_size = 600
        
        self.time = data[:, 0]
        self.time_interpolated = self.interpolate_time(target_size=target_data_size)
        self.num_data_interpolated = self.time_interpolated.shape[0]
        
        self.qpos = np.zeros((self.num_data, 35))
        self.qpos[:, 0:3] = data[:, 5:8]
        self.qpos[:, 3:7] = data[:, 1:5]
        self.qpos[:, 7:10] = data[:, 8:11]
        self.qpos[:, 14:17] = data[:, 11:14]
        self.qpos[:, 20] = data[:, 14]
        self.qpos[:, 21:24] = data[:, 15:18]
        self.qpos[:, 28:31] = data[:, 18:21]
        self.qpos[:, 34] = data[:, 21]

        print("qpos shape before interpolation:", self.qpos.shape)

        # Perform linear interpolation on self.qpos
        self.interpolate_qpos(target_size=target_data_size)
        print("qpos shape after interpolation:", self.qpos.shape)

        self.qvel = np.zeros((self.num_data_interpolated, 32))
        self.calculate_qvel()

        #self.mpos = np.zeros((self.num_data, 10))
        self.mpos = np.zeros((self.num_data_interpolated, 10))
        self.mpos[:, 0:3] = self.qpos[:, 7:10]
        self.mpos[:, 3] = self.qpos[:, 14]
        self.mpos[:, 4] = self.qpos[:, 20]
        self.mpos[:, 5:8] = self.qpos[:, 21:24]
        self.mpos[:, 8] = self.qpos[:, 28]
        self.mpos[:, 9] = self.qpos[:, 34]
        #self.mvel = np.zeros((self.num_data, 10))
        self.mvel = np.zeros((self.num_data_interpolated, 10))
        self.mvel[:, 0:3] = self.qvel[:, 6:9]
        self.mvel[:, 3] = self.qvel[:, 12]
        self.mvel[:, 4:8] = self.qvel[:, 18:22]
        self.mvel[:, 8] = self.qvel[:, 25]
        self.mvel[:, 9] = self.qvel[:, 31]
        
        self.torque = np.zeros((self.num_data, 10))
        self.calculate_torque()
        
        self.arm_pos = data[:, -6:]

    def calculate_qvel(self):
        qpos_mapped = np.zeros(shape=(self.num_data_interpolated, 32))
        qpos_mapped[:, 0:3] = self.qpos[:, 0:3]
        qpos_mapped[:, 6:9] = self.qpos[:, 7:10]
        qpos_mapped[:, 12:15] = self.qpos[:, 14:17]
        qpos_mapped[:, 18:22] = self.qpos[:, 20:24]
        qpos_mapped[:, 25:27] = self.qpos[:, 28:30]
        qpos_mapped[:, 27] = self.qpos[:, 30]
        qpos_mapped[:, 31] = self.qpos[:, 34]

        qvel = []
        for i in range(1, self.num_data_interpolated):
            qvel.append((qpos_mapped[i] - qpos_mapped[i - 1]) / (self.time_interpolated[i] - self.time_interpolated[i - 1]))
        qvel.append(qvel[-1] + (qvel[-1] - qvel[-2]) / (self.time_interpolated[-1] - self.time_interpolated[-2]))
        self.qvel = np.array(qvel)
        print("qvel shape: ", self.qvel.shape)
        
    def calculate_torque(self, kp=600, kd=600):
        self.torque = kp * self.mpos + kd * self.mvel
        print("torque shape: ", self.torque.shape)

    def interpolate_time(self, target_size=600):
        observed_time = []
        for i in range(self.num_data - 1):
            observed_time.append(self.time[i])
            num_new_col = (int)((target_size - 1) / (self.num_data - 1)) - 1
            if ((target_size - 1) % (self.num_data - 1) != 0):
                num_new_col += 1
                target_size -= 1

            for j in range(num_new_col):
                observed_time.append(observed_time[-1] + (self.time[i + 1] - self.time[i]) / (num_new_col + 1))
        observed_time.append(self.time[-1])
        return np.array(observed_time)

    def interpolate_qpos(self, target_size=600):
        num_new_col = (int)(target_size / self.num_data) - 1
        num_dimensions = self.qpos.shape[1]
        qpos_interpolated = np.zeros(shape=(self.num_data_interpolated, num_dimensions))

        for i in range(num_dimensions):
            qpos_interpolated[:, i] = np.interp(self.time_interpolated, self.time, self.qpos[:, i])
        
        self.qpos = qpos_interpolated
        
if __name__ == '__main__':
    cassie_traj = CassieTrajectory("stepdata.bin")
    cassie_traj.read_csv("out1.csv")
    