
from os import error
import torch
from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    def __init__(self, samples_list):
        self.data = samples_list

    def __len__(self):
        return len(self.data)

    def add_samples(self, samples_list):
        print(samples_list[0])
        self.data.append(samples_list)

    def __getitem__(self, idx):
        states = self.data[idx]['states']
        obs = self.data[idx]['obs']
        acts = self.data[idx]['acts']

        return torch.tensor(states), torch.tensor(obs), torch.tensor(acts)
    




        

