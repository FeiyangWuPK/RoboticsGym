import numpy as np
import abc


class TrajectoryAccumulator(abc.ABC):
    """
    Trajectory accumulator
    """
    def __init__(self, num_envs, num_obs, num_acts):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_acts = num_acts
        
        self.partial_trajectories = {env_id: [] for env_id in range(self.num_envs)}


    def add_step(self, env_id, step_dict):
        required_keys = {'obs', 'acts', 'rews', 'dones'}

        assert all(key in step_dict for key in required_keys)
        assert len(step_dict.keys()) == len(required_keys)

        self.partial_trajectories[env_id].append(step_dict)


    def get_partial_trajectories(self):
        return self.partial_trajectories
        

    def finish_trajectory(self, env_id):
        part_dicts = self.partial_trajectories[env_id]
        del self.partial_trajectories[env_id]

        return part_dicts
    

    def add_steps_and_auto_finish(self, obs, acts, rews, dones):

        trajectories = []
        for env_idx in range(self.num_envs):
            assert env_idx in self.partial_trajectories.keys()


        for env_idx, (ob, act, rew, done) in enumerate(zip(obs, acts, rews, dones)):

            self.add_step(env_idx,
                        dict(obs=ob,
                            acts=act,
                            rews=rew,
                            dones=done,
                            ))
                
            if done:
                trajectory = self.finish_trajectory(env_idx)
                trajectories.append(trajectory)

        return trajectories


                
            
                

            

                
            




        

        