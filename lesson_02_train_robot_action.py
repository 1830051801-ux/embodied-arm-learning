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


def make_training_data(num_samples=6000):
    # Observation:
    # [object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box]
    #
    # Action:
    # [move_dx, move_dy, gripper_width]
    #
    # This script creates "expert" data with a simple rule first.
    # Later, your real robot data will replace this generated expert data.
    object_xy = torch.empty(num_samples, 2).uniform_(-0.6, 0.6)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.6, 0.6)
    class_id = torch.randint(0, len(CLASS_NAMES), (num_samples,))
    class_one_hot = F.one_hot(class_id, num_classes=len(CLASS_NAMES)).float()

    obs = torch.cat([object_xy, hand_xy, class_one_hot], dim=1)

    # Expert rule:
    # Move part of the distance from current hand position to object position.
    move_xy = 0.45 * (object_xy - hand_xy)

    # Different objects need different gripper openings.
    # cup: medium, bottle: narrow, box: wide.
    gripper_width_table = torch.tensor([0.060, 0.035, 0.090])
    gripper_width = gripper_width_table[class_id].reshape(-1, 1)

    action = torch.cat([move_xy, gripper_width], dim=1)
    return obs, action


def main():
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    obs, action = make_training_data()
    train_count = int(len(obs) * 0.8)

    train_obs = obs[:train_count].to(device)
    train_action = action[:train_count].to(device)
    val_obs = obs[train_count:].to(device)
    val_action = action[train_count:].to(device)

    model = RobotPolicy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1001):
        pred = model(train_obs)
        loss = loss_fn(pred, train_action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                val_loss = loss_fn(model(val_obs), val_action)
            print(
                f"epoch={epoch:4d} "
                f"train_loss={loss.item():.8f} "
                f"val_loss={val_loss.item():.8f}"
            )

    policy_path = Path(__file__).with_name("robot_policy.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": CLASS_NAMES,
        },
        policy_path,
    )
    print("saved:", policy_path)

    # One readable test.
    # Object: bottle at (0.30, 0.10)
    # Hand: currently at (0.00, 0.00)
    test_obs = torch.tensor([[0.30, 0.10, 0.00, 0.00, 0.0, 1.0, 0.0]], device=device)
    with torch.no_grad():
        test_action = model(test_obs).cpu()[0]

    print("test case: object=bottle object_xy=(0.30, 0.10) hand_xy=(0.00, 0.00)")
    print(
        "model action: "
        f"move_dx={test_action[0]:.4f}, "
        f"move_dy={test_action[1]:.4f}, "
        f"gripper_width={test_action[2]:.4f}"
    )
    print("meaning: move toward the bottle and use a narrow gripper opening")


if __name__ == "__main__":
    main()

