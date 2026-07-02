import torch
from torch import nn
from torch.nn import functional as F


CLASS_NAMES = ["cup", "bottle", "box"]


class TinyRobotPolicy(nn.Module):
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


def make_data(num_samples=1000):
    object_xy = torch.empty(num_samples, 2).uniform_(-0.6, 0.6)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.6, 0.6)
    class_id = torch.randint(0, len(CLASS_NAMES), (num_samples,))
    class_one_hot = F.one_hot(class_id, num_classes=len(CLASS_NAMES)).float()

    obs = torch.cat([object_xy, hand_xy, class_one_hot], dim=1)

    move_xy = 0.45 * (object_xy - hand_xy)
    gripper_table = torch.tensor([0.060, 0.035, 0.090])
    gripper_width = gripper_table[class_id].reshape(-1, 1)
    action = torch.cat([move_xy, gripper_width], dim=1)

    return obs, action, class_id


def show_one_sample(obs, action, class_id, index=0):
    row = obs[index]
    act = action[index]
    name = CLASS_NAMES[int(class_id[index])]
    print("one training sample:")
    print(f"  object_type: {name}")
    print(f"  object_x/object_y: {row[0].item():.3f}, {row[1].item():.3f}")
    print(f"  hand_x/hand_y:     {row[2].item():.3f}, {row[3].item():.3f}")
    print(f"  one_hot:           {row[4:].tolist()}")
    print(f"  correct action:    move_dx={act[0].item():.3f}, move_dy={act[1].item():.3f}, gripper={act[2].item():.3f}")


def main():
    torch.manual_seed(21)

    print_header("Step 0: From Day 1 to Day 2")
    print("Day 1: x -> model -> y")
    print("Day 2: robot state -> policy model -> robot action")

    print_header("Step 1: Define the robot state")
    print("Observation columns:")
    print("  object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box")
    print("Action columns:")
    print("  move_dx, move_dy, gripper_width")

    print_header("Step 2: Create teaching data")
    obs, action, class_id = make_data()
    print("obs shape:", tuple(obs.shape))
    print("action shape:", tuple(action.shape))
    show_one_sample(obs, action, class_id, index=3)

    print_header("Step 3: Choose device and create model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = TinyRobotPolicy().to(device)
    print(model)

    obs = obs.to(device)
    action = action.to(device)

    print_header("Step 4: Test before training")
    with torch.no_grad():
        before = model(obs[:1]).cpu()[0]
    print(
        "model output before training: "
        f"move_dx={before[0]:.4f}, move_dy={before[1]:.4f}, gripper={before[2]:.4f}"
    )
    print(
        "correct action:              "
        f"move_dx={action[0, 0].item():.4f}, move_dy={action[0, 1].item():.4f}, gripper={action[0, 2].item():.4f}"
    )

    print_header("Step 5: Train state -> action")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(501):
        pred = model(obs)
        loss = loss_fn(pred, action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch in [0, 1, 2, 5, 10, 50, 100, 200, 500]:
            print(f"epoch={epoch:3d} loss={loss.item():.8f}")

    print_header("Step 6: Test after training")
    with torch.no_grad():
        after = model(obs[:1]).cpu()[0]
    correct = action[:1].cpu()[0]
    print(
        "model output after training:  "
        f"move_dx={after[0]:.4f}, move_dy={after[1]:.4f}, gripper={after[2]:.4f}"
    )
    print(
        "correct action:               "
        f"move_dx={correct[0]:.4f}, move_dy={correct[1]:.4f}, gripper={correct[2]:.4f}"
    )
    print(
        "absolute error:               "
        f"move_dx={abs(after[0]-correct[0]):.4f}, "
        f"move_dy={abs(after[1]-correct[1]):.4f}, "
        f"gripper={abs(after[2]-correct[2]):.4f}"
    )

    print_header("Step 7: Meaning for your robot arm")
    print("YOLO gives object_type and image bbox.")
    print("Calibration converts bbox center to object_x/object_y.")
    print("Robot state gives hand_x/hand_y.")
    print("PyTorch policy predicts move_dx/move_dy/gripper_width.")
    print("IK/controller converts that action into motor commands.")


if __name__ == "__main__":
    main()

