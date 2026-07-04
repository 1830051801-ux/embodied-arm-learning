from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class JointCommand:
    shoulder_deg: float
    elbow_deg: float
    gripper_width: float


def solve_reach_ik(x: float, y: float, link1: float = 0.32, link2: float = 0.28) -> tuple[float, float]:
    radius = max(1e-6, min(link1 + link2 - 1e-6, math.hypot(x, y)))
    cos_elbow = (radius * radius - link1 * link1 - link2 * link2) / (2.0 * link1 * link2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow = math.acos(cos_elbow)
    shoulder = math.atan2(y, x) - math.atan2(link2 * math.sin(elbow), link1 + link2 * math.cos(elbow))
    return math.degrees(shoulder), math.degrees(elbow)


def action_to_joint_command(hand_x: float, hand_y: float, action: list[float]) -> JointCommand:
    target_x = hand_x + action[0]
    target_y = hand_y + action[1]
    shoulder, elbow = solve_reach_ik(target_x, target_y)
    return JointCommand(shoulder_deg=shoulder, elbow_deg=elbow, gripper_width=action[2])
