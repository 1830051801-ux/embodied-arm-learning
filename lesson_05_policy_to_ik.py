import math
from pathlib import Path

import torch
from torch import nn


CLASS_NAMES = ["cup", "bottle", "box"]
LINK_1 = 0.35
LINK_2 = 0.25


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


def predict_action(model, device, object_type, object_xy, hand_xy):
    obs_values = [
        object_xy[0],
        object_xy[1],
        hand_xy[0],
        hand_xy[1],
        *one_hot_object_type(object_type),
    ]
    obs = torch.tensor([obs_values], dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(obs).cpu()[0]


def inverse_kinematics_2link(x, y, link_1=LINK_1, link_2=LINK_2):
    # Solve a planar 2-link arm.
    # Input: target end-effector position x/y.
    # Output: shoulder angle q1 and elbow angle q2 in radians.
    distance_sq = x * x + y * y
    cos_q2 = (distance_sq - link_1 * link_1 - link_2 * link_2) / (2 * link_1 * link_2)

    if cos_q2 < -1.0 or cos_q2 > 1.0:
        raise ValueError("target is outside the reachable workspace")

    q2 = math.acos(cos_q2)
    q1 = math.atan2(y, x) - math.atan2(link_2 * math.sin(q2), link_1 + link_2 * math.cos(q2))
    return q1, q2


def forward_kinematics_2link(q1, q2, link_1=LINK_1, link_2=LINK_2):
    x = link_1 * math.cos(q1) + link_2 * math.cos(q1 + q2)
    y = link_1 * math.sin(q1) + link_2 * math.sin(q1 + q2)
    return x, y


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

    # This is the kind of information YOLO + calibration would provide.
    object_type = "bottle"
    object_xy = (0.30, -0.10)

    # Current end-effector/hand position.
    hand_xy = (0.00, 0.00)

    action = predict_action(model, device, object_type, object_xy, hand_xy)
    move_dx = float(action[0])
    move_dy = float(action[1])
    gripper_width = float(action[2])

    # In this lesson, the policy output is a small movement command.
    # A real robot might use a controller to execute this as velocity,
    # or turn the new target point into joint angles with IK.
    next_hand_xy = (hand_xy[0] + move_dx, hand_xy[1] + move_dy)
    q1, q2 = inverse_kinematics_2link(next_hand_xy[0], next_hand_xy[1])
    check_xy = forward_kinematics_2link(q1, q2)
    error = math.dist(next_hand_xy, check_xy)

    print("device:", device)
    print(f"object_type={object_type}")
    print(f"object_xy=({object_xy[0]:.4f}, {object_xy[1]:.4f})")
    print(f"current_hand_xy=({hand_xy[0]:.4f}, {hand_xy[1]:.4f})")
    print(
        "pytorch_policy_output: "
        f"move_dx={move_dx:.4f}, "
        f"move_dy={move_dy:.4f}, "
        f"gripper_width={gripper_width:.4f}"
    )
    print(f"next_hand_target_xy=({next_hand_xy[0]:.4f}, {next_hand_xy[1]:.4f})")
    print(
        "ik_joint_angles: "
        f"shoulder_q1={math.degrees(q1):.2f} deg, "
        f"elbow_q2={math.degrees(q2):.2f} deg"
    )
    print(f"fk_check_xy=({check_xy[0]:.4f}, {check_xy[1]:.4f})")
    print(f"fk_error={error:.8f} m")
    print()
    print("Meaning:")
    print("1. YOLO/calibration gives object_type and object_xy.")
    print("2. PyTorch decides the next movement and gripper width.")
    print("3. IK converts the movement target into joint angles.")
    print("4. A real controller would send those angles to the motors.")


if __name__ == "__main__":
    main()

