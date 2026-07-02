import csv
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


CLASS_NAMES = ["cup", "bottle", "box"]


class SmallPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, obs):
        return self.net(obs)


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def generate_demo_rows(num_rows=12):
    torch.manual_seed(55)
    object_xy = torch.empty(num_rows, 2).uniform_(-0.5, 0.5)
    hand_xy = torch.empty(num_rows, 2).uniform_(-0.5, 0.5)
    class_id = torch.randint(0, len(CLASS_NAMES), (num_rows,))
    move_xy = 0.45 * (object_xy - hand_xy)
    gripper_table = torch.tensor([0.060, 0.035, 0.090])
    gripper_width = gripper_table[class_id]

    rows = []
    for i in range(num_rows):
        rows.append(
            {
                "object_type": CLASS_NAMES[int(class_id[i])],
                "object_x": float(object_xy[i, 0]),
                "object_y": float(object_xy[i, 1]),
                "hand_x": float(hand_xy[i, 0]),
                "hand_y": float(hand_xy[i, 1]),
                "move_dx": float(move_xy[i, 0]),
                "move_dy": float(move_xy[i, 1]),
                "gripper_width": float(gripper_width[i]),
            }
        )
    return rows


def write_csv(path, rows):
    fieldnames = [
        "object_type",
        "object_x",
        "object_y",
        "hand_x",
        "hand_y",
        "move_dx",
        "move_dy",
        "gripper_width",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_as_tensors(path):
    obs_rows = []
    action_rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = CLASS_NAMES.index(row["object_type"])
            one_hot = F.one_hot(torch.tensor(class_id), num_classes=len(CLASS_NAMES)).float()
            obs_rows.append(
                [
                    float(row["object_x"]),
                    float(row["object_y"]),
                    float(row["hand_x"]),
                    float(row["hand_y"]),
                    *one_hot.tolist(),
                ]
            )
            action_rows.append(
                [
                    float(row["move_dx"]),
                    float(row["move_dy"]),
                    float(row["gripper_width"]),
                ]
            )
    return torch.tensor(obs_rows), torch.tensor(action_rows)


def main():
    print_header("Step 0: Why data logging matters")
    print("PyTorch does not know how to grasp by itself.")
    print("It learns from many examples: robot state -> correct action.")

    print_header("Step 1: Create a tiny robot grasp CSV")
    base_dir = Path(__file__).parent
    csv_path = base_dir / "day5_demo_grasp_log.csv"
    rows = generate_demo_rows()
    write_csv(csv_path, rows)
    print("created:", csv_path)
    print("columns:", ", ".join(rows[0].keys()))

    print_header("Step 2: Look at a few rows")
    for row in rows[:3]:
        print(row)
    print("Meaning: each row is one teaching example.")

    print_header("Step 3: Convert CSV rows into tensors")
    obs, action = load_csv_as_tensors(csv_path)
    print("obs shape:", tuple(obs.shape))
    print("action shape:", tuple(action.shape))
    print("first obs:", obs[0].tolist())
    print("first action:", action[0].tolist())
    print("Meaning: obs is input, action is correct answer.")

    print_header("Step 4: Train a small policy from the CSV")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    obs = obs.to(device)
    action = action.to(device)
    model = SmallPolicy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(301):
        pred = model(obs)
        loss = loss_fn(pred, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in [0, 1, 2, 5, 10, 50, 100, 300]:
            print(f"epoch={epoch:3d} loss={loss.item():.8f}")

    print_header("Step 5: Check one prediction")
    with torch.no_grad():
        pred = model(obs[:1]).cpu()[0]
    correct = action[:1].cpu()[0]
    print(
        "prediction: "
        f"move_dx={pred[0]:.4f}, move_dy={pred[1]:.4f}, gripper={pred[2]:.4f}"
    )
    print(
        "correct:    "
        f"move_dx={correct[0]:.4f}, move_dy={correct[1]:.4f}, gripper={correct[2]:.4f}"
    )

    print_header("Step 6: Meaning for your real robot")
    print("Replace this demo CSV with your real robot log.")
    print("Log YOLO result, calibrated coordinate, robot state, action, and success.")
    print("Then train PyTorch on your own grasping data.")


if __name__ == "__main__":
    main()

