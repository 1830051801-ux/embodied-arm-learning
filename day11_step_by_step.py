import csv
import time
from pathlib import Path


FIELDNAMES = [
    "episode_id",
    "step",
    "timestamp",
    "object_type",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "object_x",
    "object_y",
    "hand_x",
    "hand_y",
    "move_dx",
    "move_dy",
    "gripper_width",
    "q1_deg",
    "q2_deg",
    "dry_run_command",
    "success",
    "failure_reason",
]


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def make_demo_episode():
    episode_id = "episode_0001"
    now = int(time.time())
    rows = [
        {
            "episode_id": episode_id,
            "step": 0,
            "timestamp": now,
            "object_type": "bottle",
            "confidence": 0.91,
            "bbox_x1": 430,
            "bbox_y1": 220,
            "bbox_x2": 500,
            "bbox_y2": 370,
            "object_x": 0.2719,
            "object_y": -0.0917,
            "hand_x": 0.0000,
            "hand_y": 0.0000,
            "move_dx": 0.1234,
            "move_dy": -0.0441,
            "gripper_width": 0.0344,
            "q1_deg": -52.39,
            "q2_deg": 163.53,
            "dry_run_command": "J -52.39 163.53 G 34.4",
            "success": "",
            "failure_reason": "",
        },
        {
            "episode_id": episode_id,
            "step": 1,
            "timestamp": now + 1,
            "object_type": "bottle",
            "confidence": 0.91,
            "bbox_x1": 430,
            "bbox_y1": 220,
            "bbox_x2": 500,
            "bbox_y2": 370,
            "object_x": 0.2719,
            "object_y": -0.0917,
            "hand_x": 0.1234,
            "hand_y": -0.0441,
            "move_dx": 0.0668,
            "move_dy": -0.0240,
            "gripper_width": 0.0344,
            "q1_deg": -45.20,
            "q2_deg": 151.10,
            "dry_run_command": "J -45.20 151.10 G 34.4",
            "success": "true",
            "failure_reason": "",
        },
    ]
    return rows


def main():
    print_header("Step 0: Why log grasp episodes?")
    print("PyTorch needs data.")
    print("A grasp log records state, action, command, and result.")

    print_header("Step 1: Create one demo episode")
    rows = make_demo_episode()
    for row in rows:
        print(row)

    print_header("Step 2: Save as CSV")
    base_dir = Path(__file__).parent
    log_path = base_dir / "day11_grasp_episode_log.csv"
    write_rows(log_path, rows)
    print("saved:", log_path)
    print("columns:")
    for name in FIELDNAMES:
        print(" ", name)

    print_header("Step 3: Explain training columns")
    print("Input to PyTorch policy:")
    print("  object_x, object_y, hand_x, hand_y, object_type")
    print("Correct action:")
    print("  move_dx, move_dy, gripper_width")
    print("Evaluation/result:")
    print("  success, failure_reason")

    print_header("Step 4: Meaning for real robot")
    print("Every real grasp attempt should append rows like this.")
    print("The CSV becomes your behavior cloning dataset.")
    print("More diverse successful data usually means a better policy.")


if __name__ == "__main__":
    main()

