# PyTorch Mechanical Arm Project Summary

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
