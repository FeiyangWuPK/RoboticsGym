from scripts.train_digit import (
    train_digit_sac,
    visualize_expert_trajectory,
    train_digit_ppo,
    train_digit_L2TRL,
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp",
    type=str,
    default="l2t",
    help="Algorithm to run. Options: ppo, sac, l2t",
)

args = parser.parse_args()


if __name__ == "__main__":

    if args.exp == "l2t":
        train_digit_L2TRL()
    elif args.exp == "ppo":
        train_digit_ppo()
    elif args.exp == "sac":
        train_digit_sac()
    elif args.exp == "visualize":
        visualize_expert_trajectory()
    else:
        raise ValueError("Invalid experiment type")
