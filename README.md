# EmbodiedArm

Vision-conditioned manipulation project for offline policy training, MuJoCo rollout checks and dry-run robot command generation.

The pipeline starts from detector output or grasp logs, converts observations into policy inputs, predicts short-horizon arm actions, validates the action in a planar MuJoCo scene, and converts the result into joint/gripper commands that can be inspected before connecting real hardware.

## Pipeline

```text
YOLO detections / grasp logs
  -> calibration adapter
  -> observation vector
  -> behavior cloning policy
  -> IK and command adapter
  -> MuJoCo rollout / dry-run command
  -> experiment metrics
```

This repository is organized as a project rather than a notebook collection. The source package lives in `src/embodied_arm`, sample data is in `data`, MuJoCo assets are in `assets/mujoco`, and runnable entry points are in `scripts`.

## Repository Layout

```text
assets/mujoco/                 MuJoCo planar arm scene
configs/default.yaml           Dataset, model and simulation settings
data/                          Sample grasp log and detector JSON
docs/                          Architecture and experiment notes
models/                        Saved policy checkpoints
scripts/                       Training, evaluation and rollout entry points
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

Train a behavior cloning policy:

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

## Current Experiment Snapshot

The included sample logs and checkpoints are small, but they keep the full path reproducible:

```text
policy checkpoint: models/robot_policy_from_csv.pt
sample grasp log: data/sample_grasp_log.csv
detector sample:  data/sample_yolo_detections.json
MuJoCo scene:     assets/mujoco/planar_grasp_scene.xml
```

The project intentionally separates three layers:

- Perception adapter: detector boxes become calibrated object positions.
- Policy layer: observation vectors become action chunks.
- Control layer: action chunks become IK and dry-run joint/gripper commands.

That separation makes it possible to replace the sample detector JSON with real YOLO output, replace synthetic logs with real teleoperation logs, or replace the planar MuJoCo model with a higher-DOF arm model without rewriting the full stack.

