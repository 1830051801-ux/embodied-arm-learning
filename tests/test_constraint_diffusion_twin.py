from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "robot_ai"))
sys.path.insert(0, str(PROJECT_DIR / "tools"))

from constraint_diffusion_twin import (
    ARCHITECTURE_NAME,
    CONTEXT_DIM,
    CONTEXT_TOKEN_NAMES,
    GRASP_INDEX,
    PLACE_INDEX,
    TRAJECTORY_STEPS,
    build_expert_trajectory,
    pose_to_context,
    rigidify_pose,
)


class ConstraintDiffusionTwinTests(unittest.TestCase):
    def test_action_chunk_transformer_contract_is_explicit(self) -> None:
        self.assertEqual(ARCHITECTURE_NAME, "embodied_action_chunk_transformer_diffusion")
        self.assertEqual(
            CONTEXT_TOKEN_NAMES,
            ("visual_grasp_pose", "visual_place_goal", "observation_uncertainty", "learned_task_token"),
        )
        self.assertEqual(CONTEXT_DIM, 21)

    def test_quintic_trajectory_preserves_keyframe_anchors(self) -> None:
        home = np.zeros(6)
        grasp = np.array([0.20, -0.10, 0.15, -0.05, 0.08, -0.03])
        place = np.array([-0.12, 0.11, -0.14, 0.06, -0.07, 0.04])
        trajectory = build_expert_trajectory(home, grasp, place)
        self.assertEqual(trajectory.shape, (TRAJECTORY_STEPS, 6))
        np.testing.assert_allclose(trajectory[0], home)
        np.testing.assert_allclose(trajectory[GRASP_INDEX], grasp)
        np.testing.assert_allclose(trajectory[PLACE_INDEX], place)
        np.testing.assert_allclose(trajectory[-1], home)

    def test_rigidify_pose_recovers_rotation_after_float32_serialization(self) -> None:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        pose[:3, 3] = [0.25, -0.12, 0.30]
        restored = rigidify_pose(pose.astype(np.float32))
        np.testing.assert_allclose(restored[:3, :3].T @ restored[:3, :3], np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(restored[:3, :3])), 1.0, places=10)
        np.testing.assert_allclose(restored[:3, 3], pose[:3, 3])

    def test_visual_context_has_stable_fixed_dimension(self) -> None:
        grasp = np.eye(4)
        place = np.eye(4)
        grasp[:3, 3] = [0.25, -0.12, 0.30]
        place[:3, 3] = [0.28, -0.08, 0.30]
        context = pose_to_context(grasp, place, 0.92, 0.003, np.deg2rad(2.0))
        self.assertEqual(context.shape, (21,))
        self.assertTrue(np.isfinite(context).all())


if __name__ == "__main__":
    unittest.main()
