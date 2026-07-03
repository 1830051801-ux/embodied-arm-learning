from __future__ import annotations

import argparse

from embodied_arm.policy import load_policy
from embodied_arm.rollout import RolloutState, rollout_step
from embodied_arm.yolo_adapter import load_best_detection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/robot_policy_from_csv.pt")
    parser.add_argument("--detections", default="data/sample_yolo_detections.json")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    policy = load_policy(args.model)
    target = load_best_detection(args.detections)
    state = RolloutState(hand_x=0.0, hand_y=0.0)
    for step in range(args.steps):
        state, command = rollout_step(policy, target, state)
        print({"step": step, "state": state, "joint_command": command})


if __name__ == "__main__":
    main()
