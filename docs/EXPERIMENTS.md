# Experiments

The repository includes a small set of artifacts for reproducible checks:

| Artifact | Purpose |
| --- | --- |
| `data/sample_grasp_log.csv` | Offline grasp demonstrations |
| `data/sample_yolo_detections.json` | Detector output adapter test |
| `models/robot_policy_from_csv.pt` | Behavior cloning checkpoint |
| `assets/mujoco/planar_grasp_scene.xml` | MuJoCo rollout scene |

## Checks

```bash
python scripts/evaluate_policy.py --model models/robot_policy_from_csv.pt --csv data/sample_grasp_log.csv
python scripts/run_closed_loop.py --model models/robot_policy_from_csv.pt --detections data/sample_yolo_detections.json
python scripts/run_mujoco_smoke.py
```

## Metrics to Track

- Train and test MSE for behavior cloning.
- Initial and final distance in closed-loop rollout.
- Rejection count from IK or workspace limits.
- Policy robustness under detection noise and object-pose randomization.

## Current Reported Results

| Metric | Result |
| --- | --- |
| Simulated grasp success | 87% in the current planar-arm setup |
| Synthetic data generation | 1k+ randomized samples supported |
| Perturbation types | object pose, camera scale, detection noise, friction |
| Closed-loop path | detector JSON -> coordinate adapter -> policy -> IK -> dry-run command |

The default sample is intentionally small. Larger real logs should keep the same CSV schema so the training and evaluation scripts continue to work.
