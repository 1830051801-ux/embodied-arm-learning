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


def make_test_data(num_samples=1000):
    torch.manual_seed(99)
    object_xy = torch.empty(num_samples, 2).uniform_(-0.6, 0.6)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.6, 0.6)
    class_id = torch.randint(0, len(CLASS_NAMES), (num_samples,))
    class_one_hot = F.one_hot(class_id, num_classes=len(CLASS_NAMES)).float()

    obs = torch.cat([object_xy, hand_xy, class_one_hot], dim=1)
    move_xy = 0.45 * (object_xy - hand_xy)
    gripper_width_table = torch.tensor([0.060, 0.035, 0.090])
    gripper_width = gripper_width_table[class_id].reshape(-1, 1)
    action = torch.cat([move_xy, gripper_width], dim=1)
    return obs, action, class_id


def evaluate_gripper_by_class(pred_action, true_action, class_id):
    for idx, name in enumerate(CLASS_NAMES):
        mask = class_id == idx
        pred_width = pred_action[mask, 2]
        true_width = true_action[mask, 2]
        mean_error = torch.mean(torch.abs(pred_width - true_width)).item()
        mean_pred = torch.mean(pred_width).item()
        mean_true = torch.mean(true_width).item()
        print(
            f"{name:6s} gripper_width "
            f"pred_mean={mean_pred:.4f} "
            f"true_mean={mean_true:.4f} "
            f"mean_abs_error={mean_error:.6f}"
        )


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

    obs, true_action, class_id = make_test_data()
    obs = obs.to(device)
    true_action = true_action.to(device)

    with torch.no_grad():
        pred_action = model(obs)

    abs_error = torch.abs(pred_action - true_action)
    mean_error = torch.mean(abs_error, dim=0)
    max_error = torch.max(abs_error, dim=0).values
    mse = torch.mean((pred_action - true_action) ** 2).item()

    print("device:", device)
    print("test_samples:", len(obs))
    print(f"mse={mse:.8f}")
    print(
        "mean_abs_error: "
        f"move_dx={mean_error[0].item():.6f}, "
        f"move_dy={mean_error[1].item():.6f}, "
        f"gripper_width={mean_error[2].item():.6f}"
    )
    print(
        "max_abs_error: "
        f"move_dx={max_error[0].item():.6f}, "
        f"move_dy={max_error[1].item():.6f}, "
        f"gripper_width={max_error[2].item():.6f}"
    )

    print()
    evaluate_gripper_by_class(pred_action.cpu(), true_action.cpu(), class_id)

    print()
    print("How to read this:")
    print("1. Lower mean_abs_error means the policy is closer to the expert actions.")
    print("2. Gripper width should be different for cup, bottle, and box.")
    print("3. If test error is much worse than train error, the model may not generalize.")


if __name__ == "__main__":
    main()

