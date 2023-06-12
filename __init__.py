from gymnasium.envs.registration import register

register(id='Cassie-v1',
		entry_point='cassie:CassieEnv',
		max_episode_steps=1000,
		autoreset=True,)