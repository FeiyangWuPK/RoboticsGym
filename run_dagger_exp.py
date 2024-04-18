from scripts.train_dagger_mujoco import (
    train_dagger,
)


if __name__ == "__main__":
    train_dagger("HalfCheetah-v4", 4, 8000)
