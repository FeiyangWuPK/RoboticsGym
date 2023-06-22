import pandas as pd
import random
import numpy as np

class DigitTrajectory:
    def __init__(self, filepath):
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
        
        self.num_data = position_full.shape[0]
        
        self.qpos = np.zeros((self.num_data, 30))
        self.qvel = np.zeros((self.num_data, 30))
        self.torque = torque