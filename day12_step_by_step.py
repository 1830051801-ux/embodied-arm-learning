import csv
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


CLASS_NAMES = ["cup", "bottle", "box"]


class LogPolicy(nn.Module):
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


def ensure_log_exists(path):
    if path.exists():
        return
    raise SystemExit("day11_grasp_episode_log.csv not found. Run day11_step_by_step.py first.")


def load_success_rows(path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row["success"]).lower() == "true":
                rows.append(row)
    return rows


def rows_to_tensors(rows):
    obs_rows = []
    action_rows = []
    for row in rows:
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
    log_path = base_dir / "day11_grasp_episode_log.csv"
    policy_path = base_dir / "day12_policy_from_log.pt"

    print_header("Step 0: Load grasp log")
    ensure_log_exists(log_path)
    print("log file:", log_path)

    print_header("Step 1: Keep successful rows")
    rows = load_success_rows(log_path)
    print("successful rows:", len(rows))
    if not rows:
        raise SystemExit("No success=true rows found. Add successful grasp rows first.")
    for row in rows:
        print(row)
    print("Meaning: first behavior cloning version learns from successful examples.")

    print_header("Step 2: Convert rows to tensors")
    obs, action = rows_to_tensors(rows)
    print("obs shape:", tuple(obs.shape))
    print("action shape:", tuple(action.shape))
    print("first obs:", obs[0].tolist())
    print("first action:", action[0].tolist())

    print_header("Step 3: Train policy from log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    obs = obs.to(device)
    action = action.to(device)
    model = LogPolicy().to(device)
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

    print_header("Step 4: Save and test policy")
    torch.save({"model_state": model.state_dict(), "class_names": CLASS_NAMES}, policy_path)
    print("saved:", policy_path)

    with torch.no_grad():
        pred = model(obs[:1]).cpu()[0]
    target = action[:1].cpu()[0]
    print(
        "policy prediction: "
        f"move_dx={pred[0]:.4f}, move_dy={pred[1]:.4f}, gripper={pred[2]:.4f}"
    )
    print(
        "logged action:      "
        f"move_dx={target[0]:.4f}, move_dy={target[1]:.4f}, gripper={target[2]:.4f}"
    )

    print_header("Step 5: Meaning")
    print("This is the smallest version of training from your own robot data.")
    print("With only one successful row it can memorize, not generalize.")
    print("Real training needs many diverse successful grasp logs.")


if __name__ == "__main__":
    main()

