import torch


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    print_header("Step 0: This course is for your robot arm project")
    print("Goal: learn PyTorch through a real robot pipeline, not abstract API memorizing.")
    print("Pipeline: YOLO -> calibration -> PyTorch policy -> IK -> dry-run command -> logs")

    print_header("Step 1: Check PyTorch")
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device used today:", device)
    print("Meaning: PyTorch is installed and can run model training code.")

    print_header("Step 2: What YOLO gives you")
    yolo_detection = {
        "class_name": "cup",
        "confidence": 0.92,
        "bbox_xyxy": [300, 180, 380, 300],
    }
    x1, y1, x2, y2 = yolo_detection["bbox_xyxy"]
    pixel_center = [(x1 + x2) / 2, (y1 + y2) / 2]
    print("YOLO detection:", yolo_detection)
    print("bbox center pixel:", pixel_center)
    print("Meaning: YOLO sees the object, but this is still image-pixel information.")

    print_header("Step 3: Calibration turns image pixels into robot coordinates")
    robot_xy = [0.42, -0.08]
    print("pixel center:", pixel_center)
    print("robot table coordinate after calibration:", robot_xy)
    print("Meaning: the arm cannot use pixel 340,240 directly; it needs robot coordinates.")

    print_header("Step 4: PyTorch policy input and output")
    obs = torch.tensor(
        [
            robot_xy[0],
            robot_xy[1],
            0.30,  # hand_x
            -0.20,  # hand_y
            1.0,  # cup one-hot flag
            0.0,  # box one-hot flag
        ],
        dtype=torch.float32,
        device=device,
    )
    action = torch.tensor(
        [
            0.12,  # move_dx
            0.12,  # move_dy
            0.04,  # gripper width
        ],
        dtype=torch.float32,
        device=device,
    )
    print("observation tensor:", obs.detach().cpu().tolist())
    print("target action tensor:", action.detach().cpu().tolist())
    print("Meaning: robot learning is usually training observation -> action.")

    print_header("Step 5: Why train instead of only writing rules")
    print("Rule example: if object is at x=0.42, move hand right by 0.12")
    print("Learning example: after many logs, model learns different moves for cup/box/edge cases")
    print("Meaning: rules are a useful starting point; data lets the system handle more variation.")

    print_header("Step 6: Safety order")
    print("1. Learn the code with fake tensors")
    print("2. Test in simulation or simple scripts")
    print("3. Print dry-run commands")
    print("4. Only then send commands to the real arm")
    print("Meaning: never let an unverified model directly drive real motors.")

    print_header("Today summary")
    print("YOLO answers: what and where in the image")
    print("Calibration answers: where on the robot table")
    print("PyTorch policy answers: what action should the arm take")
    print("IK/control answers: how each joint or motor should move")


if __name__ == "__main__":
    main()

