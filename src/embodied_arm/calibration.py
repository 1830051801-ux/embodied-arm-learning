from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AffineCalibration:
    scale_x: float = 1.0 / 640.0
    scale_y: float = 1.0 / 480.0
    offset_x: float = -0.5
    offset_y: float = -0.5

    def pixel_to_robot(self, u: float, v: float) -> tuple[float, float]:
        return u * self.scale_x + self.offset_x, v * self.scale_y + self.offset_y


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5

