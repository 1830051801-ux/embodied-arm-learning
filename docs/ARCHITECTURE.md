# Architecture

EmbodiedArm is split into four layers.

## Perception

`embodied_arm.yolo_adapter` loads detector JSON and selects the highest-confidence object. `embodied_arm.calibration` maps the bounding-box center into a robot-frame coordinate using a compact affine calibration. The adapter is intentionally small so it can be replaced by a live YOLO process or a camera-calibrated homography.

## Dataset

`embodied_arm.dataset` reads grasp logs with this schema:

```text
object_type, object_x, object_y, hand_x, hand_y, move_dx, move_dy, gripper_width
```

The observation vector is:

```text
[object_x, object_y, hand_x, hand_y, one_hot(object_type)]
```

The action vector is:

```text
[move_dx, move_dy, gripper_width]
```

## Policy

`embodied_arm.policy.BCPolicy` is a compact MLP behavior cloning policy. `scripts/train_policy.py` builds a checkpoint from grasp logs; `scripts/evaluate_policy.py` checks it against held-out samples.

The current implementation is small enough to inspect, but the package boundary supports replacing it with action chunking, diffusion policy, ACT-style sequence prediction, or a VLA command adapter later.

## Simulation and Control

`assets/mujoco/arm6_grasp_scene.xml` defines a 6-DoF tabletop arm, table, target block, distractor object, camera and position actuators. `embodied_arm.sim_env.ArmManipulationSim` uses MuJoCo when available and falls back to kinematic state updates when MuJoCo is not installed.

`embodied_arm.ik` converts Cartesian actions to joint angles. `embodied_arm.controller` builds dry-run commands for inspection before hardware connection.
