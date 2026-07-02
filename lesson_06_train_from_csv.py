import csv
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


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


def make_demo_csv(csv_path, num_rows=3000):
    torch.manual_seed(11)
    object_xy = torch.empty(num_rows, 2).uniform_(-0.6, 0.6)
    hand_xy = torch.empty(num_rows, 2).uniform_(-0.6, 0.6)
    class_id = torch.randint(0, len(CLASS_NAMES), (num_rows,))

    move_xy = 0.45 * (object_xy - hand_xy)
    gripper_width_table = torch.tensor([0.060, 0.035, 0.090])
    gripper_width = gripper_width_table[class_id]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "object_type",
                "object_x",
                "object_y",
                "hand_x",
                "hand_y",
                "move_dx",
                "move_dy",
                "gripper_width",
            ]
        )
        for i in range(num_rows):
            writer.writerow(
                [
                    CLASS_NAMES[int(class_id[i])],
                    float(object_xy[i, 0]),
                    float(object_xy[i, 1]),
                    float(hand_xy[i, 0]),
                    float(hand_xy[i, 1]),
                    float(move_xy[i, 0]),
                    float(move_xy[i, 1]),
                    float(gripper_width[i]),
                ]
            )


def load_csv(csv_path):
    obs_rows = []
    action_rows = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
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
    base_dir = Path(__file__).parent
    csv_path = base_dir / "robot_grasp_log_demo.csv"
    policy_path = base_dir / "robot_policy_from_csv.pt"

    make_demo_csv(csv_path)
    print("created demo robot log:", csv_path)

    obs, action = load_csv(csv_path)
    print("loaded rows:", len(obs))
    print("obs columns: object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box")
    print("action columns: move_dx, move_dy, gripper_width")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs = obs.to(device)
    action = action.to(device)

    model = RobotPolicy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(801):
        pred = model(obs)
        loss = loss_fn(pred, action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch={epoch:4d} loss={loss.item():.8f}")

    torch.save({"model_state": model.state_dict(), "class_names": CLASS_NAMES}, policy_path)
    print("saved policy:", policy_path)

    # Test one row, similar to a real YOLO + calibration result.
    test_object_type = "box"
    test_obs = torch.tensor(
        [[-0.20, 0.25, 0.05, -0.10, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        test_action = model(test_obs).cpu()[0]

    print()
    print("test input: object_type=box object_xy=(-0.20, 0.25) hand_xy=(0.05, -0.10)")
    print(
        "predicted action: "
        f"move_dx={test_action[0]:.4f}, "
        f"move_dy={test_action[1]:.4f}, "
        f"gripper_width={test_action[2]:.4f}"
    )
    print("meaning: this is how a logged grasp dataset becomes a trained PyTorch policy")


if __name__ == "__main__":
    main()

