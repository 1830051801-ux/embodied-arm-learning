import torch
from torch import nn


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    print_header("Step 0: What are we learning?")
    print("Goal: train a tiny model to learn y = 2x + 1")
    print("Robot version later: train a policy to learn state -> action")

    print_header("Step 1: Choose CPU or GPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("Meaning: cuda means PyTorch is using your NVIDIA GPU.")

    print_header("Step 2: Prepare training data")
    x = torch.linspace(-5, 5, 11).reshape(-1, 1)
    y = 2 * x + 1
    print("x shape:", tuple(x.shape))
    print("y shape:", tuple(y.shape))
    print("first 5 x values:", x[:5].reshape(-1).tolist())
    print("first 5 y answers:", y[:5].reshape(-1).tolist())
    print("Meaning: x is input, y is the correct answer.")

    x = x.to(device)
    y = y.to(device)

    print_header("Step 3: Create a model")
    model = nn.Linear(1, 1).to(device)
    print("model:", model)
    print("initial weight w:", model.weight.item())
    print("initial bias b:", model.bias.item())
    print("Meaning: the model starts with random w and b, so it is wrong.")

    print_header("Step 4: See the model before training")
    test_x = torch.tensor([[10.0]], device=device)
    with torch.no_grad():
        before = model(test_x).item()
    print("input x=10")
    print("model output before training:", before)
    print("correct answer:", 21)

    print_header("Step 5: Define loss and optimizer")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    print("loss_fn: MSELoss compares prediction and correct answer")
    print("optimizer: SGD changes w and b to reduce loss")

    print_header("Step 6: Train")
    for epoch in range(501):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch in [0, 1, 2, 5, 10, 50, 100, 200, 500]:
            print(
                f"epoch={epoch:3d} "
                f"loss={loss.item():.8f} "
                f"w={model.weight.item():.4f} "
                f"b={model.bias.item():.4f}"
            )

    print_header("Step 7: Test after training")
    with torch.no_grad():
        after = model(test_x).item()
    print("learned weight w:", model.weight.item())
    print("learned bias b:", model.bias.item())
    print("input x=10")
    print("model output after training:", after)
    print("correct answer:", 21)

    print_header("Step 8: Translate this to robot learning")
    print("Today: x -> model -> y")
    print("Robot: object_type + object_xy + hand_xy -> policy -> move_dx/move_dy/gripper")
    print("Same training logic, bigger input and output.")


if __name__ == "__main__":
    main()

