from pathlib import Path


PROJECT_SUMMARY = """# PyTorch Mechanical Arm Project Summary

## One-sentence pitch

Built a small embodied-AI mechanical-arm learning pipeline that connects object detection, camera calibration, PyTorch behavior cloning, inverse kinematics, dry-run robot commands, data logging, and baseline evaluation.

## System pipeline

```text
camera
-> YOLO detector
-> detection adapter
-> pixel-to-robot calibration
-> PyTorch policy
-> inverse kinematics
-> dry-run serial command
-> grasp logging
-> policy training and evaluation
```

## What each module does

```text
YOLO: detects object class and bbox.
Adapter: converts raw detector output to standard JSON.
Calibration: maps bbox center pixel to robot x/y coordinates.
PyTorch policy: predicts move_dx, move_dy, and gripper_width from robot state.
IK: converts target hand position to joint angles.
Dry-run serial layer: formats safe robot commands without moving hardware.
Logger: records state, action, command, and success/failure.
Evaluator: compares learned policy against a hand-written rule baseline.
```

## PyTorch learning formulation

Input observation:

```text
object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box
```

Output action:

```text
move_dx, move_dy, gripper_width
```

Training objective:

```text
behavior cloning: minimize prediction error between policy action and logged/expert action
```

## Baseline comparison

Rule baseline:

```text
always move 45 percent toward the object
```

Learned policy:

```text
learns object-dependent behavior from data
```

Observed teaching-demo result:

```text
rule movement error: about 0.027
learned movement error: about 0.0036
```

## Interview explanation

I first built a rule-based vision-to-grasp pipeline using YOLO, calibration, and IK. That made the system runnable and debuggable. Then I added data logging so each grasp attempt records perception, calibrated coordinates, robot state, action, IK result, command, and success/failure. With those logs, I trained a PyTorch behavior-cloning policy to predict robot actions from state. Finally, I compared the learned policy against a rule baseline to check whether the model actually improved the action prediction.

## Honest scope

This is not a full VLA model yet. It is a practical foundation for embodied learning:

```text
perception -> state -> action -> command -> log -> train -> evaluate
```

## Next upgrades

```text
1. Replace fake detections with live camera YOLO.
2. Use real calibration points from the robot workspace.
3. Confirm serial protocol and keep low-speed dry-run tests.
4. Collect many real grasp episodes.
5. Train policy on real logs and compare success rate against rule baseline.
6. Add simulation with MuJoCo/Isaac Sim for Real2Sim2Real.
7. Upgrade policy architecture toward ACT/DP/VLA-style models.
```
"""


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    print_header("Step 0: Why summarize the project?")
    print("A project is not just code. You need to explain the engineering logic and evidence.")

    print_header("Step 1: Generate project summary")
    output_path = Path(__file__).with_name("PROJECT_SUMMARY.md")
    output_path.write_text(PROJECT_SUMMARY, encoding="utf-8")
    print("saved:", output_path)

    print_header("Step 2: Interview pitch")
    print(
        "I built a vision-to-action mechanical-arm pipeline with YOLO, calibration, "
        "PyTorch behavior cloning, IK, dry-run command safety, data logging, and baseline evaluation."
    )

    print_header("Step 3: What not to overclaim")
    print("Do not say: I trained a full VLA model.")
    print("Say: I built the data and policy-learning foundation needed before VLA-scale training.")

    print_header("Step 4: Next stage")
    print("Move from teaching data to live camera, real calibration, real logs, and success-rate evaluation.")


if __name__ == "__main__":
    main()

