from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

OBJECT_TYPES = ("box", "cup", "bottle", "pen")


def object_one_hot(name: str) -> list[float]:
    values = [0.0] * len(OBJECT_TYPES)
    try:
        values[OBJECT_TYPES.index(name)] = 1.0
    except ValueError:
        pass
    return values


@dataclass(frozen=True)
class GraspSample:
    object_type: str
    object_x: float
    object_y: float
    hand_x: float
    hand_y: float
    move_dx: float
    move_dy: float
    gripper_width: float

    def observation(self) -> list[float]:
        return [
            self.object_x,
            self.object_y,
            self.hand_x,
            self.hand_y,
            *object_one_hot(self.object_type),
        ]

    def action(self) -> list[float]:
        return [self.move_dx, self.move_dy, self.gripper_width]


def load_grasp_csv(path: str | Path) -> list[GraspSample]:
    rows: list[GraspSample] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                GraspSample(
                    object_type=row["object_type"],
                    object_x=float(row["object_x"]),
                    object_y=float(row["object_y"]),
                    hand_x=float(row["hand_x"]),
                    hand_y=float(row["hand_y"]),
                    move_dx=float(row["move_dx"]),
                    move_dy=float(row["move_dy"]),
                    gripper_width=float(row["gripper_width"]),
                )
            )
    return rows


def split_samples(samples: list[GraspSample], train_ratio: float = 0.8) -> tuple[list[GraspSample], list[GraspSample]]:
    cut = max(1, min(len(samples), int(len(samples) * train_ratio)))
    return samples[:cut], samples[cut:]

