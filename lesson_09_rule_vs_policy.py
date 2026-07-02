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


def make_harder_test_data(num_samples=2000):
    torch.manual_seed(123)
    object_xy = torch.empty(num_samples, 2).uniform_(-0.6, 0.6)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.6, 0.6)
    class_id = torch.randint(0, len(CLASS_NAMES), (num_samples,))
    class_one_hot = F.one_hot(class_id, num_classes=len(CLASS_NAMES)).float()

    obs = torch.cat([object_xy, hand_xy, class_one_hot], dim=1)

    # This expert is intentionally more detailed than a simple rule:
    # cup moves moderately, bottle moves more cautiously, box moves more directly.
    move_gain_table = torch.tensor([0.45, 0.35, 0.55])
    gain = move_gain_table[class_id].reshape(-1, 1)
    move_xy = gain * (object_xy - hand_xy)

    gripper_width_table = torch.tensor([0.060, 0.035, 0.090])
    gripper_width = gripper_width_table[class_id].reshape(-1, 1)
    action = torch.cat([move_xy, gripper_width], dim=1)
    return obs, action, class_id


def simple_rule_action(obs):
    object_xy = obs[:, 0:2]
    hand_xy = obs[:, 2:4]
    class_one_hot = obs[:, 4:7]
    class_id = torch.argmax(class_one_hot, dim=1)

    # A beginner engineering rule:
    # always move 45 percent toward the object.
    # gripper width still uses object type.
    move_xy = 0.45 * (object_xy - hand_xy)
    gripper_width_table = torch.tensor([0.060, 0.035, 0.090], device=obs.device)
    gripper_width = gripper_width_table[class_id].reshape(-1, 1)
    return torch.cat([move_xy, gripper_width], dim=1)


def train_policy_for_harder_data(device):
    train_obs, train_action, _ = make_harder_test_data(num_samples=5000)
    train_obs = train_obs.to(device)
    train_action = train_action.to(device)

    model = RobotPolicy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(801):
        pred = model(train_obs)
        loss = loss_fn(pred, train_action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def summarize_error(name, pred, target):
    abs_error = torch.abs(pred - target)
    mean_error = torch.mean(abs_error, dim=0)
    mse = torch.mean((pred - target) ** 2).item()
    print(
        f"{name:14s} "
        f"mse={mse:.8f} "
        f"mean_abs(move_dx={mean_error[0].item():.5f}, "
        f"move_dy={mean_error[1].item():.5f}, "
        f"gripper={mean_error[2].item():.5f})"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("Training a policy on a harder expert rule...")
    model = train_policy_for_harder_data(device)
    model.eval()

    test_obs, test_action, class_id = make_harder_test_data(num_samples=2000)
    test_obs = test_obs.to(device)
    test_action = test_action.to(device)

    with torch.no_grad():
        rule_pred = simple_rule_action(test_obs)
        policy_pred = model(test_obs)

    print()
    summarize_error("simple_rule", rule_pred, test_action)
    summarize_error("pytorch_policy", policy_pred, test_action)

    print()
    print("Per-object movement error:")
    for idx, class_name in enumerate(CLASS_NAMES):
        mask = class_id.to(device) == idx
        rule_err = torch.mean(torch.abs(rule_pred[mask, 0:2] - test_action[mask, 0:2])).item()
        policy_err = torch.mean(torch.abs(policy_pred[mask, 0:2] - test_action[mask, 0:2])).item()
        print(f"{class_name:6s} simple_rule={rule_err:.5f} pytorch_policy={policy_err:.5f}")

    policy_path = Path(__file__).with_name("robot_policy_harder.pt")
    torch.save({"model_state": model.state_dict(), "class_names": CLASS_NAMES}, policy_path)
    print()
    print("saved:", policy_path)
    print()
    print("Meaning:")
    print("YOLO + simple rules can be the first working version.")
    print("PyTorch becomes useful when object type, position, history, and success data make the best action different from simple rules.")


if __name__ == "__main__":
    main()

