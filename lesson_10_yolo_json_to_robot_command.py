import argparse
import json
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


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pixel_to_robot_xy(pixel_xy, image_size):
    px, py = pixel_xy
    width, height = image_size
    robot_x = (px / width) * 1.20 - 0.60
    robot_y = 0.40 - (py / height) * 0.80
    return robot_x, robot_y


def one_hot_object_type(name):
    values = [0.0, 0.0, 0.0]
    values[CLASS_NAMES.index(name)] = 1.0
    return values


def predict_policy_action(model, device, object_type, object_xy, hand_xy):
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
    distance_sq = x * x + y * y
    cos_q2 = (distance_sq - link_1 * link_1 - link_2 * link_2) / (2 * link_1 * link_2)
    if cos_q2 < -1.0 or cos_q2 > 1.0:
        raise ValueError("target is outside the reachable workspace")

    q2 = math.acos(cos_q2)
    q1 = math.atan2(y, x) - math.atan2(link_2 * math.sin(q2), link_1 + link_2 * math.cos(q2))
    return math.degrees(q1), math.degrees(q2)


def clamp(value, low, high):
    return max(low, min(high, value))


def make_robot_command(q1_deg, q2_deg, gripper_width):
    safe_q1 = clamp(q1_deg, -90.0, 90.0)
    safe_q2 = clamp(q2_deg, 0.0, 170.0)
    safe_gripper_mm = clamp(gripper_width * 1000.0, 20.0, 100.0)
    return f"J {safe_q1:.2f} {safe_q2:.2f} G {safe_gripper_mm:.1f}"


def choose_detection(detections):
    supported = [d for d in detections if d["class_name"] in CLASS_NAMES]
    if not supported:
        raise ValueError("no supported detection class found")
    return max(supported, key=lambda d: d["confidence"])


def load_policy(device):
    policy_path = Path(__file__).with_name("robot_policy.pt")
    if not policy_path.exists():
        raise SystemExit(
            "robot_policy.pt not found. Run lesson_02_train_robot_action.py first."
        )

    checkpoint = torch.load(policy_path, map_location=device)
    model = RobotPolicy().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detections",
        default=str(Path(__file__).with_name("sample_yolo_detections.json")),
        help="Path to YOLO detection JSON file.",
    )
    args = parser.parse_args()

    detection_path = Path(args.detections)
    data = json.loads(detection_path.read_text(encoding="utf-8"))
    image_size = tuple(data["image_size"])
    detection = choose_detection(data["detections"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_policy(device)

    hand_xy = (0.00, 0.00)
    center_px = bbox_center(detection["bbox"])
    object_xy = pixel_to_robot_xy(center_px, image_size)
    action = predict_policy_action(model, device, detection["class_name"], object_xy, hand_xy)

    move_dx = float(action[0])
    move_dy = float(action[1])
    gripper_width = float(action[2])
    next_hand_xy = (hand_xy[0] + move_dx, hand_xy[1] + move_dy)

    q1_deg, q2_deg = inverse_kinematics_2link(next_hand_xy[0], next_hand_xy[1])
    command = make_robot_command(q1_deg, q2_deg, gripper_width)

    print("DRY RUN ONLY - no serial command is sent")
    print("device:", device)
    print("detections_file:", detection_path)
    print("chosen_detection:", detection)
    print(f"bbox_center_px=({center_px[0]:.1f}, {center_px[1]:.1f})")
    print(f"object_xy=({object_xy[0]:.4f}, {object_xy[1]:.4f})")
    print(
        "pytorch_policy_output: "
        f"move_dx={move_dx:.4f}, "
        f"move_dy={move_dy:.4f}, "
        f"gripper_width={gripper_width:.4f}"
    )
    print(f"next_hand_xy=({next_hand_xy[0]:.4f}, {next_hand_xy[1]:.4f})")
    print(f"ik_angles_deg: q1={q1_deg:.2f}, q2={q2_deg:.2f}")
    print("dry_run_robot_command:", command)
    print()
    print("To connect real YOLO, write the same JSON fields from your detector.")


if __name__ == "__main__":
    main()

