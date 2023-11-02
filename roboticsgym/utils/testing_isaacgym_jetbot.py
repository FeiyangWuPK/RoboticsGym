from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th

import gymnasium
from gymnasium import spaces
import numpy as np
import math
import carb


class JetBotEnv(gymnasium.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=256,
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp(
            {"headless": self.headless, "anti_aliasing": 0}
        )
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
        from omni.isaac.wheeled_robots.robots import WheeledRobot

        self._my_world = World(
            physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0
        )
        self._my_world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self.jetbot = self._my_world.scene.add(
            WheeledRobot(
                prim_path="/jetbot",
                name="my_jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0, 0.0, 0.03]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.jetbot_controller = DifferentialController(
            name="simple_control", wheel_radius=0.0325, wheel_base=0.1125
        )
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([0.60, 0.30, 0.05]),
                size=0.1,
                color=np.array([1.0, 0, 0]),
            )
        )
        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gymnasium.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=float("inf"), high=float("inf"), shape=(16,), dtype=np.float32
        )

        self.max_velocity = 1
        self.max_angular_velocity = math.pi
        self.reset_counter = 0
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        previous_jetbot_position, _ = self.jetbot.get_world_pose()
        # action forward velocity , angular velocity on [-1, 1]
        raw_forward = action[0]
        raw_angular = action[1]

        # we want to force the jetbot to always drive forward
        # so we transform to [0,1].  we also scale by our max velocity
        forward = (raw_forward + 1.0) / 2.0
        forward_velocity = forward * self.max_velocity

        # we scale the angular, but leave it on [-1,1] so the
        # jetbot can remain an ambiturner.
        angular_velocity = raw_angular * self.max_angular_velocity

        # we apply our actions to the jetbot
        for i in range(self._skip_frame):
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(
                    command=[forward_velocity, angular_velocity]
                )
            )
            self._my_world.step(render=False)

        observations = self.get_observations()
        info = {}
        done = False
        truncated = False
        if (
            self._my_world.current_time_step_index - self._steps_after_reset
            >= self._max_episode_length
        ):
            done = True
            truncated = True
        goal_world_position, _ = self.goal.get_world_pose()
        current_jetbot_position, _ = self.jetbot.get_world_pose()
        previous_dist_to_goal = np.linalg.norm(
            goal_world_position - previous_jetbot_position
        )
        current_dist_to_goal = np.linalg.norm(
            goal_world_position - current_jetbot_position
        )
        reward = previous_dist_to_goal - current_dist_to_goal
        if current_dist_to_goal < 0.1:
            done = True
        return observations, reward, done, truncated, info

    def reset(self, seed=None):
        self._my_world.reset()
        self.reset_counter = 0
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 1.00 * math.sqrt(np.random.rand()) + 0.20
        self.goal.set_world_pose(
            np.array([math.sin(alpha) * r, math.cos(alpha) * r, 0.05])
        )
        observations = self.get_observations()
        return observations, {}

    def get_observations(self):
        self._my_world.render()
        jetbot_world_position, jetbot_world_orientation = self.jetbot.get_world_pose()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        goal_world_position, _ = self.goal.get_world_pose()
        obs = np.concatenate(
            [
                jetbot_world_position,
                jetbot_world_orientation,
                jetbot_linear_velocity,
                jetbot_angular_velocity,
                goal_world_position,
            ]
        )
        return obs

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]


if __name__ == "__main__":
    # Log directory of the tensorboard files to visualize the training and for the final policy as well
    log_dir = "./cnn_policy"
    # set headles to False to visualize the policy while training
    my_env = JetBotEnv(headless=True)

    # policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[128, 128, 128], vf=[128, 128, 128])])  # Policy params
    # policy = CnnPolicy
    total_timesteps = 500000

    # Saves a checkpoint policy in the same log directory
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=log_dir, name_prefix="jetbot_policy_checkpoint"
    )
    # PPO algorithm params
    model = PPO(
        "MlpPolicy",
        my_env,
        # policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2560,
        batch_size=64,
        learning_rate=0.000125,
        gamma=0.9,
        ent_coef=7.5e-08,
        clip_range=0.3,
        n_epochs=5,
        device="cuda",
        gae_lambda=1.0,
        max_grad_norm=0.9,
        vf_coef=0.95,
        # tensorboard_log=log_dir,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(log_dir + "/jetbot_policy")  # Saves the final policy

    my_env.close()  # Closes the environment
