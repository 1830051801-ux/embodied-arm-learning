from __future__ import annotations

import csv
import math
from pathlib import Path

from .domain_randomization import sample_scene


def generate_synthetic_grasp_log(path: str | Path, samples: int = 1000, seed: int = 7) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["object_type", "object_x", "object_y", "hand_x", "hand_y", "move_dx", "move_dy", "gripper_width"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(samples):
            scene = sample_scene(seed + i)
            hand_x = 0.0
            hand_y = 0.0
            dx = scene.object_x - hand_x
            dy = scene.object_y - hand_y
            norm = max(1e-6, math.hypot(dx, dy))
            step = min(0.28, norm)
            writer.writerow(
                {
                    "object_type": "box" if i % 3 == 0 else "cup" if i % 3 == 1 else "bottle",
                    "object_x": scene.object_x,
                    "object_y": scene.object_y,
                    "hand_x": hand_x,
                    "hand_y": hand_y,
                    "move_dx": dx / norm * step,
                    "move_dy": dy / norm * step,
                    "gripper_width": 0.09 if i % 3 == 0 else 0.06,
                }
            )

