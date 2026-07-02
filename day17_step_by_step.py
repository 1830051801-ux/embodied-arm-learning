import torch
from torch import nn


class CurrentOnlyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, obs):
        return self.net(obs)


class HistoryPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, obs):
        return self.net(obs)


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def make_dataset(num_samples=6000, seed=601):
    torch.manual_seed(seed)
    prev_target = torch.empty(num_samples, 2).uniform_(-0.45, 0.45)
    velocity = torch.empty(num_samples, 2).uniform_(-0.06, 0.06)
    current_target = prev_target + velocity
    next_target = current_target + velocity
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.45, 0.45)

    # The expert reaches toward the predicted next target, not just the current target.
    action = 0.45 * (next_target - hand_xy)

    current_obs = torch.cat([current_target, hand_xy], dim=1)
    history_obs = torch.cat([prev_target, current_target, hand_xy], dim=1)
    return current_obs, history_obs, action


def train_model(model, obs, action, device):
    model = model.to(device)
    obs = obs.to(device)
    action = action.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(501):
        pred = model(obs)
        loss = loss_fn(pred, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def evaluate(name, model, obs, action, device):
    model.eval()
    with torch.no_grad():
        pred = model(obs.to(device)).cpu()
    err = torch.mean(torch.abs(pred - action), dim=0)
    mse = torch.mean((pred - action) ** 2).item()
    print(f"{name:14s} mse={mse:.8f} mean_abs_dx={err[0].item():.5f} mean_abs_dy={err[1].item():.5f}")
    return pred


def main():
    print_header("Step 0: Why temporal context?")
    print("If a target is moving, one frame gives position but not velocity.")
    print("Two frames let the policy infer movement direction.")

    print_header("Step 1: Create moving-target dataset")
    train_current, train_history, train_action = make_dataset(6000, seed=601)
    test_current, test_history, test_action = make_dataset(2000, seed=602)
    print("current_obs shape:", tuple(train_current.shape))
    print("history_obs shape:", tuple(train_history.shape))
    print("action shape:", tuple(train_action.shape))
    print("current obs: [current_target_x, current_target_y, hand_x, hand_y]")
    print("history obs: [prev_target_x, prev_target_y, current_target_x, current_target_y, hand_x, hand_y]")

    print_header("Step 2: Train current-only policy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    current_model = train_model(CurrentOnlyPolicy(), train_current, train_action, device)

    print_header("Step 3: Train history policy")
    history_model = train_model(HistoryPolicy(), train_history, train_action, device)

    print_header("Step 4: Compare on test data")
    evaluate("current_only", current_model, test_current, test_action, device)
    evaluate("history", history_model, test_history, test_action, device)

    print_header("Step 5: Inspect one example")
    idx = 0
    prev_target = test_history[idx, 0:2]
    current_target = test_history[idx, 2:4]
    hand_xy = test_history[idx, 4:6]
    velocity = current_target - prev_target
    next_target = current_target + velocity
    print(f"prev_target=({prev_target[0]:+.4f}, {prev_target[1]:+.4f})")
    print(f"current_target=({current_target[0]:+.4f}, {current_target[1]:+.4f})")
    print(f"estimated_velocity=({velocity[0]:+.4f}, {velocity[1]:+.4f})")
    print(f"predicted_next_target=({next_target[0]:+.4f}, {next_target[1]:+.4f})")
    print(f"hand_xy=({hand_xy[0]:+.4f}, {hand_xy[1]:+.4f})")

    print_header("Step 6: Meaning")
    print("History gives the policy motion information.")
    print("This is why robot policies often consume observation sequences, not just one state.")


if __name__ == "__main__":
    main()

