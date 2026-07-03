from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RandomizedScene:
    object_x: float
    object_y: float
    camera_scale: float
    detection_noise_px: float
    friction: float


def sample_scene(seed: int | None = None) -> RandomizedScene:
    rng = random.Random(seed)
    return RandomizedScene(
        object_x=rng.uniform(0.18, 0.48),
        object_y=rng.uniform(-0.22, 0.22),
        camera_scale=rng.uniform(0.96, 1.04),
        detection_noise_px=rng.uniform(0.0, 3.0),
        friction=rng.uniform(0.55, 1.15),
    )

