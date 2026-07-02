import torch
from torch import nn


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


class CalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, pixel_xy_normalized):
        return self.net(pixel_xy_normalized)


def true_pixel_to_robot(pixel_xy):
    # Hidden ground-truth mapping used to create teaching data.
    # In a real robot project, robot_xy comes from measured calibration points.
    px = pixel_xy[:, 0]
    py = pixel_xy[:, 1]
    robot_x = (px / 640.0) * 1.20 - 0.60
    robot_y = 0.40 - (py / 480.0) * 0.80
    return torch.stack([robot_x, robot_y], dim=1)


def normalize_pixel(pixel_xy):
    scale = torch.tensor([640.0, 480.0], device=pixel_xy.device)
    return pixel_xy / scale


def main():
    torch.manual_seed(88)

    print_header("Step 0: Why calibration?")
    print("YOLO gives pixel coordinates.")
    print("The robot needs robot workspace coordinates.")
    print("Calibration learns or computes pixel_xy -> robot_xy.")

    print_header("Step 1: Create calibration points")
    pixel_xy = torch.tensor(
        [
            [80.0, 80.0],
            [320.0, 80.0],
            [560.0, 80.0],
            [80.0, 240.0],
            [320.0, 240.0],
            [560.0, 240.0],
            [80.0, 400.0],
            [320.0, 400.0],
            [560.0, 400.0],
        ]
    )
    robot_xy = true_pixel_to_robot(pixel_xy)
    for i in range(len(pixel_xy)):
        print(
            f"point {i}: pixel=({pixel_xy[i,0]:.0f}, {pixel_xy[i,1]:.0f}) "
            f"robot=({robot_xy[i,0]:.4f}, {robot_xy[i,1]:.4f})"
        )
    print("Meaning: each row is a measured pixel-to-robot calibration pair.")

    print_header("Step 2: Train a tiny PyTorch calibration model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    pixel_xy = pixel_xy.to(device)
    robot_xy = robot_xy.to(device)
    pixel_norm = normalize_pixel(pixel_xy)

    model = CalibrationModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    for epoch in range(1001):
        pred_robot_xy = model(pixel_norm)
        loss = loss_fn(pred_robot_xy, robot_xy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in [0, 1, 2, 5, 10, 50, 100, 300, 1000]:
            print(f"epoch={epoch:4d} calibration_loss={loss.item():.8f}")

    print_header("Step 3: Test on a YOLO bbox center")
    bbox = [430, 220, 500, 370]
    center_px = torch.tensor([[(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]], device=device)
    true_robot = true_pixel_to_robot(center_px.cpu()).to(device)
    with torch.no_grad():
        pred_robot = model(normalize_pixel(center_px))
    error = torch.linalg.norm(pred_robot - true_robot, dim=1).item()
    print("bbox:", bbox)
    print(f"center_px=({center_px[0,0].item():.1f}, {center_px[0,1].item():.1f})")
    print(f"pred_robot_xy=({pred_robot[0,0].item():.4f}, {pred_robot[0,1].item():.4f})")
    print(f"true_robot_xy=({true_robot[0,0].item():.4f}, {true_robot[0,1].item():.4f})")
    print(f"calibration_error={error:.6f} m")

    print_header("Step 4: Meaning for your real project")
    print("Replace these fake calibration pairs with measured points on your table.")
    print("Your real YOLO bbox center goes through calibration before PyTorch policy.")
    print("If calibration is wrong, the arm will reach the wrong place even if YOLO is correct.")


if __name__ == "__main__":
    main()

