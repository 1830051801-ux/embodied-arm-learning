from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .calibration import AffineCalibration, bbox_center


@dataclass(frozen=True)
class DetectionTarget:
    object_type: str
    confidence: float
    object_x: float
    object_y: float


def load_best_detection(path: str | Path, calibration: AffineCalibration | None = None) -> DetectionTarget:
    calibration = calibration or AffineCalibration()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    detections = data.get("detections", [])
    if not detections:
        raise ValueError("no detections in file")
    best = max(detections, key=lambda item: float(item.get("confidence", 0.0)))
    u, v = bbox_center([float(x) for x in best["bbox"]])
    x, y = calibration.pixel_to_robot(u, v)
    return DetectionTarget(
        object_type=str(best["class_name"]),
        confidence=float(best["confidence"]),
        object_x=x,
        object_y=y,
    )

