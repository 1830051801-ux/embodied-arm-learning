import torch
from torch import nn


IMAGE_SIZE = 16


class VisionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(16 * IMAGE_SIZE * IMAGE_SIZE + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, image, hand_xy):
        visual_feature = self.cnn(image)
        x = torch.cat([visual_feature, hand_xy], dim=1)
        return self.head(x)


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def make_dataset(num_samples=4000, seed=901):
    torch.manual_seed(seed)
    pixel_xy = torch.randint(0, IMAGE_SIZE, (num_samples, 2))
    image = torch.zeros(num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)
    for i in range(num_samples):
        px = int(pixel_xy[i, 0])
        py = int(pixel_xy[i, 1])
        image[i, 0, py, px] = 1.0

    # Convert pixel position to robot-like normalized coordinate.
    object_x = (pixel_xy[:, 0].float() / (IMAGE_SIZE - 1)) * 1.0 - 0.5
    object_y = 0.5 - (pixel_xy[:, 1].float() / (IMAGE_SIZE - 1)) * 1.0
    object_xy = torch.stack([object_x, object_y], dim=1)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.5, 0.5)
    action = 0.45 * (object_xy - hand_xy)
    return image, hand_xy, action, object_xy


def main():
    print_header("Step 0: Vision to action")
    print("Instead of giving object_x/object_y directly, we give a small image.")
    print("A CNN extracts visual features, then the policy predicts movement.")

    print_header("Step 1: Create tiny image dataset")
    image, hand_xy, action, object_xy = make_dataset()
    print("image shape:", tuple(image.shape))
    print("hand_xy shape:", tuple(hand_xy.shape))
    print("action shape:", tuple(action.shape))
    print("one object_xy:", object_xy[0].tolist())
    print("one hand_xy:", hand_xy[0].tolist())
    print("one action:", action[0].tolist())

    print_header("Step 2: Train CNN policy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    image = image.to(device)
    hand_xy = hand_xy.to(device)
    action = action.to(device)
    model = VisionPolicy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(501):
        pred = model(image, hand_xy)
        loss = loss_fn(pred, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in [0, 10, 50, 100, 300, 500]:
            print(f"epoch={epoch:3d} loss={loss.item():.8f}")

    print_header("Step 3: Test on one image")
    test_image, test_hand, test_action, test_object = make_dataset(num_samples=1, seed=902)
    test_image = test_image.to(device)
    test_hand = test_hand.to(device)
    with torch.no_grad():
        pred = model(test_image, test_hand).cpu()[0]
    target = test_action[0]
    print("test object_xy hidden from model:", test_object[0].tolist())
    print("test hand_xy:", test_hand.cpu()[0].tolist())
    print(f"pred action=({pred[0].item():+.4f}, {pred[1].item():+.4f})")
    print(f"true action=({target[0].item():+.4f}, {target[1].item():+.4f})")

    print_header("Step 4: Meaning")
    print("YOLO is one way to turn images into object coordinates.")
    print("CNN/Vision encoder is another way to feed image information into a policy.")
    print("VLA uses much stronger vision-language encoders, but the input-output idea is the same.")


if __name__ == "__main__":
    main()

