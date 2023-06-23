import pandas as pd
import random
import numpy as np

class DigitTrajectory:
    def __init__(self, filepath):
        self.dim_qpos = 61
        self.dim_qvel = 54
        self.read_csv(filepath)
        
    def state(self, t):
        i = int(t % self.num_data)
        return (self.qpos[i], self.qvel[i])

    def action(self, t):
        i = int(t % self.num_data)
        return (self.mpos[i], self.mvel[i], self.torque[i])

    def sample(self):
        i = random.randrange(self.num_data)
        return (self.time[i], self.qpos[i], self.qvel[i])
    
    def read_csv(self, filepath):
        
        # Read in the recorded data.
        digit_state_distill = pd.read_csv(filepath)
        # Extract the position, velocity, and torque.
        
        # The definition of position_full and velocity_full is in digit_main. 
        # https://github.gatech.edu/GeorgiaTechLIDARGroup/digit_main/blob/d11392ff2c08593005b2d5e0187e3e9c0fd84f49/include/digit_definition.hpp#L17
        position_full = digit_state_distill.loc[:,'position_full_0':'position_full_29'].to_numpy()

        velocity_full = digit_state_distill.loc[:,'velocity_full_0':'velocity_full_29'].to_numpy()

        # The definition of motor torque is also in digit_main.
        # https://github.gatech.edu/GeorgiaTechLIDARGroup/digit_main/blob/d11392ff2c08593005b2d5e0187e3e9c0fd84f49/include/digit_definition.hpp#L100
        torque = digit_state_distill.loc[:,'torque_0':'torque_19'].to_numpy()
        
        base_position = digit_state_distill.loc[:,'base_position_0':'base_position_2'].to_numpy()
        
        base_velocity = digit_state_distill.loc[:,'base_velocity_0':'base_velocity_2'].to_numpy()
        
        self.num_data = position_full.shape[0]
        
        self.qpos = np.zeros((self.num_data, self.dim_qpos))
        self.qpos[:, 0:3] = base_position[:, 0:3]
        self.qpos[:, 2] += np.ones(self.num_data) * 1.06 # Add a constant height.
        self.qpos[:, 3:7] = np.tile(np.array([1, 0, 0, 0]), (self.num_data,1))
        
        self.qpos[:, 7:10] = position_full[:, 0:3]
        self.qpos[:, 14] = position_full[:, 3] # left-knee
        self.qpos[:, 15:17] = position_full[:, 20:22]
        self.qpos[:, 17] = position_full[:, 24]
        self.qpos[:, 18] = position_full[:, 4]
        self.qpos[:, 23] = position_full[:, 5]
        self.qpos[:, 28:30] = position_full[:, 22:24]
        self.qpos[:, 30:34] = position_full[:, 12:16]
        
        self.qpos[:, 34:37] = position_full[:, 6:9]
        self.qpos[:, 41] = position_full[:, 9] # right-knee
        self.qpos[:, 42:44] = position_full[:, 25:27]
        self.qpos[:, 44] = position_full[:, 29]
        self.qpos[:, 45] = position_full[:, 10]
        self.qpos[:, 50] = position_full[:, 11]
        self.qpos[:, 55:57] = position_full[:, 27:29]
        self.qpos[:, 57:61] = position_full[:, 16:20]
        
        self.qvel = np.zeros((self.num_data, self.dim_qvel))
        self.qvel[:, 0:3] = base_velocity[:, 0:3]
        self.qvel[:, 3:6] = np.tile(np.array([0, 0, 0]), (self.num_data,1))
        
        self.qvel[:, 6:9] = velocity_full[:, 0:3]
        self.qvel[:, 12] = velocity_full[:, 3] # left-knee
        self.qvel[:, 13:15] = velocity_full[:, 20:22]
        self.qvel[:, 15] = velocity_full[:, 24]
        self.qvel[:, 16] = velocity_full[:, 4]
        self.qvel[:, 20] = velocity_full[:, 5]
        self.qvel[:, 24:26] = velocity_full[:, 22:24]
        self.qvel[:, 26:30] = velocity_full[:, 12:16]
        
        self.qvel[:, 30:33] = velocity_full[:, 6:9]
        self.qvel[:, 36] = velocity_full[:, 9] # right-knee
        self.qvel[:, 37:39] = velocity_full[:, 25:27]
        self.qvel[:, 39] = velocity_full[:, 29]
        self.qvel[:, 40] = velocity_full[:, 10]
        self.qvel[:, 44] = velocity_full[:, 11]
        self.qvel[:, 48:50] = velocity_full[:, 27:29]
        self.qvel[:, 50:54] = velocity_full[:, 16:20]

        self.torque = torque
        
        