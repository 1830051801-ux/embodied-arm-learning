from __future__ import annotations

from pathlib import Path

from .dataset import load_grasp_csv, split_samples
from .policy import BCPolicy

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None
    TensorDataset = None


def train_behavior_cloning(
    csv_path: str | Path,
    output_path: str | Path,
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("PyTorch is required for training")

    samples = load_grasp_csv(csv_path)
    train_samples, test_samples = split_samples(samples)
    x_train = torch.tensor([s.observation() for s in train_samples], dtype=torch.float32)
    y_train = torch.tensor([s.action() for s in train_samples], dtype=torch.float32)
    x_test = torch.tensor([s.observation() for s in test_samples], dtype=torch.float32)
    y_test = torch.tensor([s.action() for s in test_samples], dtype=torch.float32)

    policy = BCPolicy(obs_dim=x_train.shape[1], action_dim=y_train.shape[1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for obs, action in loader:
            pred = policy(obs)
            loss = torch.nn.functional.mse_loss(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    policy.eval()
    with torch.no_grad():
        train_mse = torch.nn.functional.mse_loss(policy(x_train), y_train).item()
        test_mse = torch.nn.functional.mse_loss(policy(x_test), y_test).item() if len(test_samples) else train_mse

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output_path)
    return {"train_mse": train_mse, "test_mse": test_mse, "samples": float(len(samples))}

