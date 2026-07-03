from __future__ import annotations

from dataclasses import dataclass

from .dataset import object_one_hot
from .ik import JointCommand, action_to_joint_command
from .policy import BCPolicy, predict_action
from .yolo_adapter import DetectionTarget


@dataclass
class RolloutState:
    hand_x: float
    hand_y: float


def build_observation(target: DetectionTarget, state: RolloutState) -> list[float]:
    return [target.object_x, target.object_y, state.hand_x, state.hand_y, *object_one_hot(target.object_type)]


def rollout_step(policy: BCPolicy, target: DetectionTarget, state: RolloutState) -> tuple[RolloutState, JointCommand]:
    action = predict_action(policy, build_observation(target, state))
    command = action_to_joint_command(state.hand_x, state.hand_y, action)
    next_state = RolloutState(hand_x=state.hand_x + action[0], hand_y=state.hand_y + action[1])
    return next_state, command

