from scripts.train_digit import (
    train_digit_sac,
    visualize_expert_trajectory,
    train_digit_ppo,
    train_digit_L2TRL,
)

if __name__ == "__main__":
    train_digit_ppo()
    # visualize_expert_trajectory()
