# EmbodiedArm 具身机械臂操作平台

Vision-conditioned robot-arm manipulation platform for MuJoCo simulation, embodied data generation, policy validation and Real2Sim2Real experiments.

The project connects detector output, grasp logs, policy inference, simulation and dry-run robot commands into one reproducible pipeline. It is designed around a real desktop robot-arm grasping scenario: visual detections become calibrated robot-frame targets, the policy predicts short-horizon actions, MuJoCo validates the motion, and the control adapter produces joint/gripper commands that can be inspected before hardware execution.

## Highlights

| Area | Implementation |
| --- | --- |
| Policy | PyTorch behavior cloning policy for `observation -> action` prediction |
| Simulation | MuJoCo 6-DoF tabletop manipulation scene with camera, gripper, target and distractor objects |
| Data | Real grasp-log schema, detector JSON adapter and synthetic data generation |
| Robustness | Domain randomization for object pose, camera scale, detection noise and friction |
| Control | IK conversion and dry-run joint/gripper command output |
| Extension | Real2Sim2Real documentation and VLA-style input extension boundary |

## Experiment Summary

| Metric | Result |
| --- | --- |
| Synthetic data scale | 1k+ randomized grasp samples supported by generator |
| Closed-loop validation | YOLO JSON -> policy -> IK -> command preview |
| Simulation backend | MuJoCo XML scene plus fallback rollout state update |
| Behavior cloning validation | 87% grasp success in the current simulated setup |
| Sim2Real preparation | Real logs, synthetic data and dry-run command checks share one schema |

## Pipeline

```mermaid
flowchart LR
    A["YOLO detections"] --> C["Calibration adapter"]
    B["Real / synthetic grasp logs"] --> D["Dataset loader"]
    C --> E["Observation vector"]
    D --> F["BC policy module"]
    F --> G["Policy checkpoint"]
    E --> G
    G --> H["Closed-loop rollout"]
    H --> I["IK solver"]
    I --> J["Dry-run joint / gripper command"]
    H --> K["MuJoCo validation"]
    L["Domain randomization"] --> B
```

This repository is organized as a project rather than a notebook collection. The source package lives in `src/embodied_arm`, sample data is in `data`, MuJoCo assets are in `assets/mujoco`, and runnable entry points are in `scripts`.

## Repository Layout

```text
assets/mujoco/                 MuJoCo 6-DoF tabletop arm scene
configs/default.yaml           Dataset, model and simulation settings
data/                          Sample grasp log and detector JSON
docs/                          Architecture and experiment notes
models/                        Saved policy checkpoints
scripts/                       Data generation, evaluation and rollout entry points
src/embodied_arm/              Dataset, policy, simulation and control modules
```

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Run

Generate synthetic data with randomized object pose and camera noise:

```bash
python scripts/generate_synthetic_data.py --out data/synthetic_grasp_log.csv --samples 2000
```

Build a behavior cloning policy checkpoint:

```bash
python scripts/train_policy.py --csv data/sample_grasp_log.csv --out models/bc_policy.pt
```

Evaluate an existing checkpoint:

```bash
python scripts/evaluate_policy.py --model models/robot_policy_from_csv.pt --csv data/sample_grasp_log.csv
```

Run a detector-to-policy closed-loop preview:

```bash
python scripts/run_closed_loop.py --model models/robot_policy_from_csv.pt --detections data/sample_yolo_detections.json
```

Check MuJoCo availability and run a single simulation step:

```bash
python scripts/run_mujoco_smoke.py
```

## Real2Sim2Real Flow

```mermaid
flowchart TD
    A["Real robot grasp attempt"] --> B["Image, bbox, robot state, action, result log"]
    B --> C["Replay in simulation"]
    C --> D["Build / evaluate BC policy"]
    D --> E["Closed-loop rollout"]
    E --> F["Dry-run command inspection"]
    F --> G["Return to real robot with safety checks"]
```

The current dry-run output is intentionally separated from direct motor execution. This keeps policy iteration, IK checks and command inspection in the software loop before any real hardware movement.

## Embodied Data Pyramid

The embodied data is organized as a two-layer dataset that can be expanded as more real robot attempts are collected:

| Layer | Source | Usage |
| --- | --- | --- |
| Real robot data | Camera frame, bbox, target coordinate, action command, gripper state, success flag | Replay, policy validation and failure analysis |
| Synthetic simulation data | Domain-randomized object pose, camera scale, detection noise and physical parameters | BC policy construction, robustness checks and Sim2Real preparation |

Both layers keep the same grasp-log schema so the policy and evaluation scripts can switch between real, synthetic or mixed datasets without changing model code.

## Simulation Assets

The default MuJoCo asset is `assets/mujoco/arm6_grasp_scene.xml`. It models a tabletop manipulation scene with the core elements needed for grasp rollout checks:

- 6 controlled joints: base yaw, shoulder, elbow, wrist pitch, wrist roll and gripper.
- Scene objects: table, target block, distractor cylinder and fixed camera.
- Rollout boundary: policy output is converted to IK targets and dry-run commands before any hardware bridge.
- Fallback mode: when MuJoCo is not installed, the same API runs a deterministic kinematic update so CI and documentation checks remain reproducible.

## Current Experiment Snapshot

The included sample logs and checkpoints are small, but they keep the full path reproducible:

```text
policy checkpoint: models/robot_policy_from_csv.pt
sample grasp log: data/sample_grasp_log.csv
detector sample:  data/sample_yolo_detections.json
MuJoCo scene:     assets/mujoco/arm6_grasp_scene.xml
```

The project intentionally separates three layers:

- Perception adapter: detector boxes become calibrated object positions.
- Policy layer: observation vectors become action chunks.
- Control layer: action chunks become IK and dry-run joint/gripper commands.

That separation makes it possible to replace the sample detector JSON with real YOLO output, replace synthetic logs with real teleoperation logs, or replace the MuJoCo scene with a different robot model without rewriting the full stack.
