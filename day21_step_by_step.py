import torch
from torch import nn
from torch.nn import functional as F


IMAGE_SIZE = 16
COMMANDS = ["pick", "push", "pull"]


class MiniVLAPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        fused_dim = 16 * IMAGE_SIZE * IMAGE_SIZE + len(COMMANDS) + 2
        self.policy = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, image, command_one_hot, hand_xy):
        vision_feature = self.vision(image)
        fused = torch.cat([vision_feature, command_one_hot, hand_xy], dim=1)
        return self.policy(fused)


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def make_dataset(num_samples=5000, seed=1001):
    torch.manual_seed(seed)
    pixel_xy = torch.randint(0, IMAGE_SIZE, (num_samples, 2))
    image = torch.zeros(num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)
    for i in range(num_samples):
        px = int(pixel_xy[i, 0])
        py = int(pixel_xy[i, 1])
        image[i, 0, py, px] = 1.0

    object_x = (pixel_xy[:, 0].float() / (IMAGE_SIZE - 1)) * 1.0 - 0.5
    object_y = 0.5 - (pixel_xy[:, 1].float() / (IMAGE_SIZE - 1)) * 1.0
    object_xy = torch.stack([object_x, object_y], dim=1)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.5, 0.5)
    command_id = torch.randint(0, len(COMMANDS), (num_samples,))
    command_one_hot = F.one_hot(command_id, num_classes=len(COMMANDS)).float()

    to_object = object_xy - hand_xy
    action = torch.zeros(num_samples, 3)

    pick = command_id == COMMANDS.index("pick")
    action[pick, 0:2] = 0.45 * to_object[pick]
    action[pick, 2] = 0.035

    push = command_id == COMMANDS.index("push")
    action[push, 0:2] = 0.35 * to_object[push] + torch.tensor([0.060, 0.000])
    action[push, 2] = 0.080

    pull = command_id == COMMANDS.index("pull")
    action[pull, 0:2] = 0.35 * to_object[pull] + torch.tensor([-0.060, 0.000])
    action[pull, 2] = 0.055

    return image, command_one_hot, hand_xy, action, object_xy


def make_single_image(object_xy):
    px = int(round(((object_xy[0] + 0.5) / 1.0) * (IMAGE_SIZE - 1)))
    py = int(round(((0.5 - object_xy[1]) / 1.0) * (IMAGE_SIZE - 1)))
    px = max(0, min(IMAGE_SIZE - 1, px))
    py = max(0, min(IMAGE_SIZE - 1, py))
    image = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    image[0, 0, py, px] = 1.0
    return image


def command_vector(command):
    command_id = COMMANDS.index(command)
    return F.one_hot(torch.tensor([command_id]), num_classes=len(COMMANDS)).float()


def main():
    print_header("Step 0: Mini VLA")
    print("Input: image + language command + robot state.")
    print("Output: move_dx, move_dy, gripper_width.")

    print_header("Step 1: Create multimodal dataset")
    image, command_one_hot, hand_xy, action, object_xy = make_dataset()
    print("image shape:", tuple(image.shape))
    print("command_one_hot shape:", tuple(command_one_hot.shape))
    print("hand_xy shape:", tuple(hand_xy.shape))
    print("action shape:", tuple(action.shape))
    print("one object_xy hidden in image:", object_xy[0].tolist())
    print("one command_one_hot:", command_one_hot[0].tolist())
    print("one hand_xy:", hand_xy[0].tolist())
    print("one action:", action[0].tolist())

    print_header("Step 2: Train Mini VLA policy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    image = image.to(device)
    command_one_hot = command_one_hot.to(device)
    hand_xy = hand_xy.to(device)
    action = action.to(device)
    model = MiniVLAPolicy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(501):
        pred = model(image, command_one_hot, hand_xy)
        loss = loss_fn(pred, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in [0, 10, 50, 100, 300, 500]:
            print(f"epoch={epoch:3d} loss={loss.item():.8f}")

    print_header("Step 3: Same image/state, different language")
    test_object = (0.30, 0.10)
    test_hand = torch.tensor([[0.00, 0.00]], dtype=torch.float32, device=device)
    test_image = make_single_image(test_object).to(device)
    print(f"test_object={test_object}, test_hand=(0.00, 0.00)")
    for command in COMMANDS:
        cmd = command_vector(command).to(device)
        with torch.no_grad():
            pred = model(test_image, cmd, test_hand).cpu()[0]
        print(
            f"command={command:4s} "
            f"move_dx={pred[0].item():+.4f} "
            f"move_dy={pred[1].item():+.4f} "
            f"gripper={pred[2].item():.4f}"
        )

    print_header("Step 4: Meaning")
    print("Vision tells where the object is.")
    print("Language tells what task to do.")
    print("Robot state tells where the hand is.")
    print("The policy fuses all three to output action.")


if __name__ == "__main__":
    main()

