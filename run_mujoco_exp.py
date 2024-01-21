from scripts.train_newalgo import (
    run_mujoco_rl,
)

if __name__ == "__main__":
    for env_id in [
        "Ant-v4",
        "HalfCheetah-v4",
        "Hopper-v4",
        "Walker2d-v4",
        "Humanoid-v4",
    ]:
        run_mujoco_rl(env_id)
