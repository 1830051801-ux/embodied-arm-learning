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

    # Expert behavior is object-dependent.
    # A simple fixed rule cannot match all three classes.
    move_gain = torch.tensor([0.45, 0.35, 0.55])[class_id].reshape(-1, 1)
    move_xy = move_gain * (object_xy - hand_xy)
    gripper = torch.tensor([0.060, 0.035, 0.090])[class_id].reshape(-1, 1)
    action = torch.cat([move_xy, gripper], dim=1)
    return obs, action, class_id


def simple_rule(obs):
    object_xy = obs[:, 0:2]
    hand_xy = obs[:, 2:4]
    class_id = torch.argmax(obs[:, 4:7], dim=1)
    move_xy = 0.45 * (object_xy - hand_xy)
    gripper = torch.tensor([0.060, 0.035, 0.090], device=obs.device)[class_id].reshape(-1, 1)
    return torch.cat([move_xy, gripper], dim=1)


def summarize(name, pred, target):
    abs_error = torch.abs(pred - target)
    mean_error = torch.mean(abs_error, dim=0)
    mse = torch.mean((pred - target) ** 2).item()
    print(
        f"{name:14s} "
        f"mse={mse:.8f} "
        f"mean_abs_dx={mean_error[0].item():.5f} "
        f"mean_abs_dy={mean_error[1].item():.5f} "
        f"mean_abs_gripper={mean_error[2].item():.5f}"
    )


def main():
    print_header("Step 0: What are we checking?")
    print("A model is useful only if it works on new test data.")
    print("We will compare a simple rule with a PyTorch policy.")

    print_header("Step 1: Create train and test data")
    train_obs, train_action, _ = make_dataset(5000, seed=100)
    test_obs, test_action, test_class_id = make_dataset(1500, seed=200)
    print("train_obs shape:", tuple(train_obs.shape))
    print("test_obs shape:", tuple(test_obs.shape))
    print("Input columns: object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box")
    print("Output columns: move_dx, move_dy, gripper_width")

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
        train_loss = loss_fn(pred, train_action)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch in [0, 1, 2, 5, 10, 50, 100, 300, 700]:
            with torch.no_grad():
                test_loss = loss_fn(model(test_obs), test_action)
            print(f"epoch={epoch:3d} train_loss={train_loss.item():.8f} test_loss={test_loss.item():.8f}")

    print_header("Step 3: Compare simple rule vs PyTorch policy")
    with torch.no_grad():
        rule_pred = simple_rule(test_obs)
        policy_pred = model(test_obs)

    summarize("simple_rule", rule_pred, test_action)
    summarize("pytorch", policy_pred, test_action)

    print_header("Step 4: Check error by object type")
    for idx, name in enumerate(CLASS_NAMES):
        mask = test_class_id == idx
        rule_move_err = torch.mean(torch.abs(rule_pred[mask, 0:2] - test_action[mask, 0:2])).item()
        policy_move_err = torch.mean(torch.abs(policy_pred[mask, 0:2] - test_action[mask, 0:2])).item()
        print(f"{name:6s} move_error simple_rule={rule_move_err:.5f} pytorch={policy_move_err:.5f}")

    print_header("Step 5: Meaning")
    print("The simple rule is okay when its assumption matches the expert.")
    print("The PyTorch policy can learn object-dependent behavior from data.")
    print("This is why your YOLO + rule system is a good start, but not the full embodied-AI story.")


if __name__ == "__main__":
    main()

