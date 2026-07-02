import torch
from torch import nn


def main():
    # Lesson 01:
    # Learn the smallest PyTorch training loop.
    # Robot learning uses the same structure, but with bigger inputs and outputs.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Data: x is the input, y is the correct answer.
    # The hidden rule is y = 2x + 1.
    x = torch.linspace(-5, 5, 200).reshape(-1, 1).to(device)
    y = (2 * x + 1).to(device)

    # Model: one input number -> one output number.
    model = nn.Linear(1, 1).to(device)

    # Loss: how wrong the model is.
    # Optimizer: how PyTorch changes the model to reduce the loss.
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1001):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch={epoch:4d} loss={loss.item():.8f}")

    w = model.weight.item()
    b = model.bias.item()
    print(f"learned formula: y = {w:.4f} * x + {b:.4f}")

    test_x = torch.tensor([[10.0]], device=device)
    with torch.no_grad():
        test_y = model(test_x)
    print("when x=10, model output:", test_y.item())
    print("correct answer should be:", 21)


if __name__ == "__main__":
    main()

