# from scripts.train_newalgo import run_mujoco_rl, run_mujoco_second_stage
from scripts.train_lt2_mujoco import train_hopper_rl, run_hopper_second_stage, run_mujoco_rl, halfcheetah_ablation


def mujoco_rl_exp():
    for env_id in [
        # "Ant-v4",
        # "HalfCheetah-v4",
        # "Hopper-v4",
        "Walker2d-v4",
        # "Humanoid-v4",
    ]:
        run_mujoco_rl(env_id)
        run_mujoco_second_stage(env_id)


def mujoco_second_stage_exp():
    for env_id in [
        # "Ant-v4",
        # "HalfCheetah-v4",
        "Hopper-v4",
        # "Walker2d-v4",
        # "Humanoid-v4",
    ]:
        run_mujoco_second_stage(env_id)


if __name__ == "__main__":
    # mujoco_second_stage_exp()
    halfcheetah_ablation()
    # train_hopper_rl()
    # run_hopper_second_stage()
