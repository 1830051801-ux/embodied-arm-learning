from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - allows static inspection without torch installed
    torch = None
    nn = None


class BCPolicy(nn.Module if nn else object):
    def __init__(self, obs_dim: int = 8, action_dim: int = 3, hidden_dim: int = 64):
        if nn is None:
            raise RuntimeError("PyTorch is required to instantiate BCPolicy")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def load_policy(path: str | Path, obs_dim: int = 8, action_dim: int = 3) -> BCPolicy:
    if torch is None:
        raise RuntimeError("PyTorch is required to load a policy")
    policy = BCPolicy(obs_dim=obs_dim, action_dim=action_dim)
    state = torch.load(Path(path), map_location="cpu")
    policy.load_state_dict(state)
    policy.eval()
    return policy


def predict_action(policy: BCPolicy, observation: list[float]) -> list[float]:
    if torch is None:
        raise RuntimeError("PyTorch is required for policy inference")
    with torch.no_grad():
        obs = torch.tensor([observation], dtype=torch.float32)
        action = policy(obs)[0].cpu().tolist()
    return [float(x) for x in action]

