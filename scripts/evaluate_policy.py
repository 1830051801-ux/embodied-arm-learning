from __future__ import annotations

import argparse

from embodied_arm.evaluate import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/robot_policy_from_csv.pt")
    parser.add_argument("--csv", default="data/sample_grasp_log.csv")
    args = parser.parse_args()
    print(evaluate_policy(args.model, args.csv))


if __name__ == "__main__":
    main()
