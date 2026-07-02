import json
from pathlib import Path

import torch


CLASS_NAMES = ["cup", "bottle", "box"]


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pixel_to_robot_xy(pixel_xy, image_size):
    px, py = pixel_xy
    width, height = image_size

    # Teaching calibration:
    # image x: 0..640 maps to robot x: -0.60..0.60 m
    # image y: 0..480 maps to robot y:  0.40..-0.40 m
    robot_x = (px / width) * 1.20 - 0.60
    robot_y = 0.40 - (py / height) * 0.80
    return robot_x, robot_y


def one_hot(class_name):
    values = [0.0, 0.0, 0.0]
    values[CLASS_NAMES.index(class_name)] = 1.0
    return values


def main():
    print_header("Step 0: What does YOLO provide?")
    print("YOLO provides class_name, confidence, and bbox.")
    print("PyTorch policy needs numbers, not raw text or pixels.")

    print_header("Step 1: Load fake YOLO JSON")
    path = Path(__file__).with_name("sample_yolo_detections.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    print("file:", path)
    print("image_size:", data["image_size"])
    print("detections:")
    for det in data["detections"]:
        print(" ", det)

    print_header("Step 2: Choose a target detection")
    detections = [d for d in data["detections"] if d["class_name"] in CLASS_NAMES]
    target = max(detections, key=lambda d: d["confidence"])
    print("chosen target:", target)
    print("Meaning: here we choose the supported detection with highest confidence.")

    print_header("Step 3: Convert bbox to pixel center")
    center_px = bbox_center(target["bbox"])
    print("bbox:", target["bbox"])
    print(f"center_px=({center_px[0]:.1f}, {center_px[1]:.1f})")
    print("Meaning: this is still image pixel coordinate, not robot coordinate.")

    print_header("Step 4: Convert pixel center to robot xy")
    image_size = tuple(data["image_size"])
    object_xy = pixel_to_robot_xy(center_px, image_size)
    print(f"robot object_xy=({object_xy[0]:.4f}, {object_xy[1]:.4f}) meters")
    print("Meaning: calibration converts camera pixels into robot workspace coordinates.")

    print_header("Step 5: Convert class name to one-hot")
    class_vector = one_hot(target["class_name"])
    print("class_name:", target["class_name"])
    print("one_hot:", class_vector)
    print("Meaning: neural networks need numeric inputs.")

    print_header("Step 6: Build PyTorch policy input tensor")
    hand_xy = (0.00, 0.00)
    obs_values = [
        object_xy[0],
        object_xy[1],
        hand_xy[0],
        hand_xy[1],
        *class_vector,
    ]
    obs = torch.tensor([obs_values], dtype=torch.float32)
    print("hand_xy:", hand_xy)
    print("obs values:", obs_values)
    print("obs tensor shape:", tuple(obs.shape))
    print("obs tensor:", obs)

    print_header("Step 7: Meaning for your real project")
    print("Real YOLO will replace sample_yolo_detections.json.")
    print("Real calibration will replace the simple pixel_to_robot_xy function.")
    print("The final obs tensor is what your PyTorch policy uses to predict action.")


if __name__ == "__main__":
    main()

