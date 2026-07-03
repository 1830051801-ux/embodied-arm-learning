from __future__ import annotations

from dataclasses import asdict

from .ik import JointCommand


def build_dry_run_command(command: JointCommand, seq: int = 1) -> dict:
    return {
        "seq": seq,
        "cmd": "joint_grasp_preview",
        "joints_deg": {
            "shoulder": round(command.shoulder_deg, 3),
            "elbow": round(command.elbow_deg, 3),
        },
        "gripper_width_m": round(command.gripper_width, 4),
        "raw": asdict(command),
    }

