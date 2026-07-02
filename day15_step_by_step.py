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


def make_training_data(num_samples=5000):
    torch.manual_seed(401)
    object_xy = torch.empty(num_samples, 2).uniform_(-0.5, 0.5)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.5, 0.5)
    class_id = torch.randint(0, len(CLASS_NAMES), (num_samples,))
    one_hot = F.one_hot(class_id, num_classes=len(CLASS_NAMES)).float()
    obs = torch.cat([object_xy, hand_xy, one_hot], dim=1)

    move_xy = 0.35 * (object_xy - hand_xy)
    gripper = torch.tensor([0.060, 0.035, 0.090])[class_id].reshape(-1, 1)
    action = torch.cat([move_xy, gripper], dim=1)
    return obs, action


def make_obs(object_xy, hand_xy, object_type, device):
    class_id = CLASS_NAMES.index(object_type)
    one_hot = [0.0, 0.0, 0.0]
    one_hot[class_id] = 1.0
    values = [object_xy[0], object_xy[1], hand_xy[0], hand_xy[1], *one_hot]
    return torch.tensor([values], dtype=torch.float32, device=device)


def train_policy(device):
    obs, action = make_training_data()
    obs = obs.to(device)
    action = action.to(device)

    model = Policy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(501):
        pred = model(obs)
        loss = loss_fn(pred, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in [0, 50, 100, 300, 500]:
            print(f"epoch={epoch:3d} train_loss={loss.item():.8f}")
    return model


def main():
    print_header("Step 0: What is closed-loop rollout?")
    print("A policy is run repeatedly.")
    print("Each step observes current hand position and outputs the next movement.")

    print_header("Step 1: Train a simple policy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = train_policy(device)
    model.eval()

    print_header("Step 2: Roll out the policy")
    object_type = "bottle"
    object_xy = (0.35, -0.20)
    hand_xy = [0.00, 0.00]
    print(f"object_type={object_type}")
    print(f"target object_xy=({object_xy[0]:.3f}, {object_xy[1]:.3f})")
    print(f"start hand_xy=({hand_xy[0]:.3f}, {hand_xy[1]:.3f})")

    for step in range(12):
        obs = make_obs(object_xy, hand_xy, object_type, device)
        with torch.no_grad():
            action = model(obs).cpu()[0]

        move_dx = float(action[0])
        move_dy = float(action[1])
        gripper = float(action[2])
        hand_xy[0] += move_dx
        hand_xy[1] += move_dy
        distance = ((object_xy[0] - hand_xy[0]) ** 2 + (object_xy[1] - hand_xy[1]) ** 2) ** 0.5

        print(
            f"step={step:02d} "
            f"move=({move_dx:+.4f}, {move_dy:+.4f}) "
            f"hand=({hand_xy[0]:+.4f}, {hand_xy[1]:+.4f}) "
            f"distance_to_target={distance:.5f} "
            f"gripper={gripper:.4f}"
        )

    print_header("Step 3: Meaning")
    print("If distance_to_target decreases, the policy is moving the hand toward the object.")
    print("A real robot would also update hand position from sensors/FK after each command.")
    print("This is the start of closed-loop robot control.")


if __name__ == "__main__":
    main()

