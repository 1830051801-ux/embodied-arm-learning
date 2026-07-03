from __future__ import annotations

from pathlib import Path

from .dataset import load_grasp_csv
from .policy import load_policy, predict_action


def evaluate_policy(model_path: str | Path, csv_path: str | Path) -> dict[str, float]:
    samples = load_grasp_csv(csv_path)
    if not samples:
        raise ValueError("empty evaluation dataset")
    policy = load_policy(model_path)
    squared = []
    for sample in samples:
        pred = predict_action(policy, sample.observation())
        target = sample.action()
        squared.extend((p - t) ** 2 for p, t in zip(pred, target))
    mse = sum(squared) / len(squared)
    return {"mse": mse, "samples": float(len(samples))}

