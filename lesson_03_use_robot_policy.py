from pathlib import Path

import torch
from torch import nn


CLASS_NAMES = ["cup", "bottle", "box"]


class RobotPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, obs):
        return self.net(obs)


def one_hot_object_type(name):
    values = [0.0, 0.0, 0.0]
    values[CLASS_NAMES.index(name)] = 1.0
    return values


def predict(model, device, object_type, object_xy, hand_xy):
    obs_values = [
        object_xy[0],
        object_xy[1],
        hand_xy[0],
        hand_xy[1],
        *one_hot_object_type(object_type),
    ]
    obs = torch.tensor([obs_values], dtype=torch.float32, device=device)
    with torch.no_grad():
        action = model(obs).cpu()[0]
    return action


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_path = Path(__file__).with_name("robot_policy.pt")
    if not policy_path.exists():
        raise SystemExit(
            "robot_policy.pt not found. Run lesson_02_train_robot_action.py first."
        )

    checkpoint = torch.load(policy_path, map_location=device)
    model = RobotPolicy().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    cases = [
        ("cup", (0.20, 0.10), (0.00, 0.00)),
        ("bottle", (0.30, -0.15), (0.05, 0.05)),
        ("box", (-0.25, 0.20), (0.10, -0.10)),
    ]

    print("device:", device)
    for object_type, object_xy, hand_xy in cases:
        action = predict(model, device, object_type, object_xy, hand_xy)
        print()
        print(f"object_type={object_type}")
        print(f"object_xy={object_xy}, hand_xy={hand_xy}")
        print(
            "policy output: "
            f"move_dx={action[0]:.4f}, "
            f"move_dy={action[1]:.4f}, "
            f"gripper_width={action[2]:.4f}"
        )

    print()
    print("In the real robot project, YOLO and calibration provide object_type/object_xy.")
    print("The policy output is then converted to joint motion by IK or a controller.")


if __name__ == "__main__":
    main()

