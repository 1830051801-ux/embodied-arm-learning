import torch
from torch import nn
from torch.nn import functional as F


COMMANDS = ["pick", "push", "pull"]


class LanguageConditionedPolicy(nn.Module):
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


def make_dataset(num_samples=6000, seed=801):
    torch.manual_seed(seed)
    object_xy = torch.empty(num_samples, 2).uniform_(-0.5, 0.5)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.5, 0.5)
    command_id = torch.randint(0, len(COMMANDS), (num_samples,))
    command_one_hot = F.one_hot(command_id, num_classes=len(COMMANDS)).float()

    obs = torch.cat([object_xy, hand_xy, command_one_hot], dim=1)

    to_object = object_xy - hand_xy
    action = torch.zeros(num_samples, 3)

    # pick: move toward object and close gripper.
    pick_mask = command_id == COMMANDS.index("pick")
    action[pick_mask, 0:2] = 0.45 * to_object[pick_mask]
    action[pick_mask, 2] = 0.035

    # push: move toward object but with a positive x bias, gripper open.
    push_mask = command_id == COMMANDS.index("push")
    action[push_mask, 0:2] = 0.35 * to_object[push_mask] + torch.tensor([0.060, 0.000])
    action[push_mask, 2] = 0.080

    # pull: move toward object but with a negative x bias, gripper half-open.
    pull_mask = command_id == COMMANDS.index("pull")
    action[pull_mask, 0:2] = 0.35 * to_object[pull_mask] + torch.tensor([-0.060, 0.000])
    action[pull_mask, 2] = 0.055

    return obs, action, command_id


def make_obs(object_xy, hand_xy, command, device):
    command_id = COMMANDS.index(command)
    one_hot = [0.0, 0.0, 0.0]
    one_hot[command_id] = 1.0
    values = [object_xy[0], object_xy[1], hand_xy[0], hand_xy[1], *one_hot]
    return torch.tensor([values], dtype=torch.float32, device=device)


def main():
    print_header("Step 0: What does language condition mean?")
    print("Same visual state can need different actions depending on instruction.")
    print("Example: pick object, push object, pull object.")

    print_header("Step 1: Create language-conditioned dataset")
    obs, action, command_id = make_dataset()
    print("obs shape:", tuple(obs.shape))
    print("action shape:", tuple(action.shape))
    print("obs columns: object_x, object_y, hand_x, hand_y, is_pick, is_push, is_pull")
    print("action columns: move_dx, move_dy, gripper_width")
    for idx, command in enumerate(COMMANDS):
        count = int((command_id == idx).sum())
        print(f"{command}: {count} samples")

    print_header("Step 2: Train language-conditioned policy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    obs = obs.to(device)
    action = action.to(device)
    model = LanguageConditionedPolicy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(701):
        pred = model(obs)
        loss = loss_fn(pred, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in [0, 10, 50, 100, 300, 700]:
            print(f"epoch={epoch:3d} loss={loss.item():.8f}")

    print_header("Step 3: Same object state, different language commands")
    object_xy = (0.30, 0.10)
    hand_xy = (0.00, 0.00)
    print(f"object_xy={object_xy}, hand_xy={hand_xy}")
    for command in COMMANDS:
        test_obs = make_obs(object_xy, hand_xy, command, device)
        with torch.no_grad():
            pred = model(test_obs).cpu()[0]
        print(
            f"command={command:4s} "
            f"move_dx={pred[0].item():+.4f} "
            f"move_dy={pred[1].item():+.4f} "
            f"gripper={pred[2].item():.4f}"
        )

    print_header("Step 4: Meaning")
    print("The policy sees the same object and hand coordinates.")
    print("The language one-hot changes the output action.")
    print("This is a tiny version of the VLA idea: vision/state + language -> action.")


if __name__ == "__main__":
    main()

