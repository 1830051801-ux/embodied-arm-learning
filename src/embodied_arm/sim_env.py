from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .ik import planar_ik


@dataclass
class SimObservation:
    object_x: float
    object_y: float
    hand_x: float
    hand_y: float
    object_type: str = "box"


@dataclass
class SimStep:
    observation: SimObservation
    reward: float
    done: bool
    distance: float


class PlanarArmSim:
    """Small MuJoCo-backed planar arm environment with a kinematic fallback."""

    def __init__(self, model_path: str | Path = "assets/mujoco/planar_grasp_scene.xml"):
        self.model_path = Path(model_path)
        self.object_x = 0.36
        self.object_y = 0.10
        self.hand_x = 0.0
        self.hand_y = 0.0
        self._mujoco = None
        self._model = None
        self._data = None
        try:
            import mujoco

            self._mujoco = mujoco
            self._model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self._data = mujoco.MjData(self._model)
        except Exception:
            self._mujoco = None

    @property
    def has_mujoco(self) -> bool:
        return self._mujoco is not None

    def reset(self, object_x: float = 0.36, object_y: float = 0.10) -> SimObservation:
        self.object_x = object_x
        self.object_y = object_y
        self.hand_x = 0.0
        self.hand_y = 0.0
        if self.has_mujoco:
            self._data.qpos[:] = 0.0
            self._data.ctrl[:] = 0.0
            self._mujoco.mj_forward(self._model, self._data)
        return self.observe()

    def observe(self) -> SimObservation:
        return SimObservation(
            object_x=self.object_x,
            object_y=self.object_y,
            hand_x=self.hand_x,
            hand_y=self.hand_y,
        )

    def step(self, action: list[float]) -> SimStep:
        self.hand_x += float(action[0])
        self.hand_y += float(action[1])
        if self.has_mujoco:
            shoulder, elbow = planar_ik(self.hand_x, self.hand_y)
            self._data.ctrl[0] = math.radians(shoulder)
            self._data.ctrl[1] = math.radians(elbow)
            for _ in range(10):
                self._mujoco.mj_step(self._model, self._data)

        distance = math.hypot(self.object_x - self.hand_x, self.object_y - self.hand_y)
        done = distance < 0.025
        reward = -distance + (1.0 if done else 0.0)
        return SimStep(observation=self.observe(), reward=reward, done=done, distance=distance)

