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


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pixel_to_robot_xy(pixel_xy, image_size):
    # Simple teaching calibration:
    # image x: 0..width  -> robot x: -0.60..0.60 meters
    # image y: 0..height -> robot y:  0.40..-0.40 meters
    #
    # In the real project, replace this with camera calibration or homography.
    px, py = pixel_xy
    width, height = image_size
    robot_x = (px / width) * 1.20 - 0.60
    robot_y = 0.40 - (py / height) * 0.80
    return robot_x, robot_y


def one_hot_object_type(name):
    values = [0.0, 0.0, 0.0]
    values[CLASS_NAMES.index(name)] = 1.0
    return values


def policy_action(model, device, object_type, object_xy, hand_xy):
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

    image_size = (640, 480)
    hand_xy = (0.00, 0.00)

    # Fake YOLO detections.
    # A real YOLO model gives class_name, confidence, and bbox.
    detections = [
        {"class_name": "bottle", "confidence": 0.91, "bbox": (430, 220, 500, 370)},
        {"class_name": "cup", "confidence": 0.86, "bbox": (250, 170, 330, 300)},
        {"class_name": "box", "confidence": 0.78, "bbox": (80, 260, 200, 420)},
    ]

    print("device:", device)
    print("image_size:", image_size)
    print("hand_xy:", hand_xy)

    for det in detections:
        class_name = det["class_name"]
        center_px = bbox_center(det["bbox"])
        object_xy = pixel_to_robot_xy(center_px, image_size)
        action = policy_action(model, device, class_name, object_xy, hand_xy)

        print()
        print("fake_yolo_detection:", det)
        print(f"bbox_center_px=({center_px[0]:.1f}, {center_px[1]:.1f})")
        print(f"calibrated_robot_xy=({object_xy[0]:.4f}, {object_xy[1]:.4f})")
        print(
            "pytorch_policy_output: "
            f"move_dx={action[0]:.4f}, "
            f"move_dy={action[1]:.4f}, "
            f"gripper_width={action[2]:.4f}"
        )
        print("next real step: convert move_dx/move_dy to joint motion by IK/controller")


if __name__ == "__main__":
    main()

