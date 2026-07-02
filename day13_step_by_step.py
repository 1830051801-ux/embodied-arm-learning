import torch
from torch import nn
from torch.nn import functional as F


CLASS_NAMES = ["cup", "bottle", "box"]


class Policy(nn.Module):
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


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def make_dataset(num_samples, seed):
    torch.manual_seed(seed)
    object_xy = torch.empty(num_samples, 2).uniform_(-0.6, 0.6)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.6, 0.6)
    class_id = torch.randint(0, len(CLASS_NAMES), (num_samples,))
    one_hot = F.one_hot(class_id, num_classes=len(CLASS_NAMES)).float()
    obs = torch.cat([object_xy, hand_xy, one_hot], dim=1)

    # Simulated expert:
    # cup: normal approach
    # bottle: cautious approach
    # box: direct approach
    gain_table = torch.tensor([0.45, 0.35, 0.55])
    gain = gain_table[class_id].reshape(-1, 1)
    move_xy = gain * (object_xy - hand_xy)
    gripper = torch.tensor([0.060, 0.035, 0.090])[class_id].reshape(-1, 1)
    action = torch.cat([move_xy, gripper], dim=1)
    return obs, action, class_id


def rule_policy(obs):
    object_xy = obs[:, 0:2]
    hand_xy = obs[:, 2:4]
    class_id = torch.argmax(obs[:, 4:7], dim=1)
    move_xy = 0.45 * (object_xy - hand_xy)
    gripper = torch.tensor([0.060, 0.035, 0.090], device=obs.device)[class_id].reshape(-1, 1)
    return torch.cat([move_xy, gripper], dim=1)


def metric(pred, target):
    abs_error = torch.abs(pred - target)
    return {
        "mse": torch.mean((pred - target) ** 2).item(),
        "move_error": torch.mean(abs_error[:, 0:2]).item(),
        "gripper_error": torch.mean(abs_error[:, 2]).item(),
    }


def main():
    print_header("Step 0: Why compare with rules?")
    print("A PyTorch policy should be compared against a simple baseline.")
    print("Otherwise you cannot prove it improves anything.")

    print_header("Step 1: Create train/test data")
    train_obs, train_action, _ = make_dataset(6000, seed=301)
    test_obs, test_action, test_class_id = make_dataset(2000, seed=302)
    print("train samples:", len(train_obs))
    print("test samples:", len(test_obs))
    print("Expert action differs by object type.")

    print_header("Step 2: Train PyTorch policy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    train_obs = train_obs.to(device)
    train_action = train_action.to(device)
    test_obs = test_obs.to(device)
    test_action = test_action.to(device)
    test_class_id = test_class_id.to(device)

    model = Policy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(701):
        pred = model(train_obs)
        loss = loss_fn(pred, train_action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in [0, 10, 50, 100, 300, 700]:
            with torch.no_grad():
                test_loss = loss_fn(model(test_obs), test_action)
            print(f"epoch={epoch:3d} train_loss={loss.item():.8f} test_loss={test_loss.item():.8f}")

    print_header("Step 3: Evaluate rule baseline vs learned policy")
    with torch.no_grad():
        rule_pred = rule_policy(test_obs)
        learned_pred = model(test_obs)

    rule_metric = metric(rule_pred, test_action)
    learned_metric = metric(learned_pred, test_action)
    print("rule baseline:", rule_metric)
    print("learned policy:", learned_metric)

    print_header("Step 4: Per-object movement error")
    for idx, name in enumerate(CLASS_NAMES):
        mask = test_class_id == idx
        rule_err = torch.mean(torch.abs(rule_pred[mask, 0:2] - test_action[mask, 0:2])).item()
        learned_err = torch.mean(torch.abs(learned_pred[mask, 0:2] - test_action[mask, 0:2])).item()
        print(f"{name:6s}: rule={rule_err:.5f}, learned={learned_err:.5f}")

    print_header("Step 5: Interview-style conclusion")
    print("Engineering first version: YOLO + calibration + rule + IK.")
    print("Embodied learning upgrade: log data, train policy, compare against rule baseline.")
    print("Claim improvement only when test metrics or real success rate improve.")


if __name__ == "__main__":
    main()

