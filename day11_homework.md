# Day 11 小作业 - 抓取日志怎么变成训练数据

今天你要搞懂这件事：

```text
没有日志，就没有你自己机械臂的数据集。
没有数据集，PyTorch 就只能学假数据或别人数据。
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 11
```

它会生成：

```text
day11_grasp_episode_log.csv
```

## 打开 CSV

打开：

```text
day11_grasp_episode_log.csv
```

你会看到这些列：

```text
episode_id
step
timestamp
object_type
confidence
bbox_x1, bbox_y1, bbox_x2, bbox_y2
object_x, object_y
hand_x, hand_y
move_dx, move_dy
gripper_width
q1_deg, q2_deg
dry_run_command
success
failure_reason
```

## 哪些列是 PyTorch 输入 obs

训练策略时，输入通常是：

```text
object_x
object_y
hand_x
hand_y
object_type
```

转换成 tensor 后类似：

```text
[object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box]
```

这些表示“当前状态是什么”。

## 哪些列是正确动作 action

正确动作通常是：

```text
move_dx
move_dy
gripper_width
```

这些表示“这个状态下应该怎么动”。

PyTorch 要学的是：

```text
obs -> action
```

## 哪些列用于评估和筛选

这些列不一定直接作为 action，但很重要：

```text
success
failure_reason
confidence
dry_run_command
q1_deg
q2_deg
```

用途：

```text
success=true             可以作为成功示范训练
success=false            用来分析失败或以后训练避错
failure_reason           帮你知道失败原因
confidence               可过滤 YOLO 置信度太低的样本
q1_deg/q2_deg            可检查 IK 和关节限制
dry_run_command          可追溯当时准备发什么命令
```

## 动手改一下：加入失败样本

打开：

```text
day11_step_by_step.py
```

在 `rows = [` 里面再加一行，表示失败抓取，比如：

```python
{
    "episode_id": episode_id,
    "step": 2,
    "timestamp": now + 2,
    "object_type": "bottle",
    "confidence": 0.91,
    "bbox_x1": 430,
    "bbox_y1": 220,
    "bbox_x2": 500,
    "bbox_y2": 370,
    "object_x": 0.2719,
    "object_y": -0.0917,
    "hand_x": 0.1902,
    "hand_y": -0.0681,
    "move_dx": 0.0000,
    "move_dy": 0.0000,
    "gripper_width": 0.0344,
    "q1_deg": -45.20,
    "q2_deg": 151.10,
    "dry_run_command": "J -45.20 151.10 G 34.4",
    "success": "false",
    "failure_reason": "missed_object",
}
```

再跑：

```powershell
.\run_day.bat 11
```

观察 CSV 是否多了一行失败记录。

## 为什么失败样本也有价值

成功样本可以教模型：

```text
什么状态下应该怎么做
```

失败样本可以帮你分析：

```text
是 YOLO 错了？
是标定错了？
是 IK 超限？
是夹爪太宽/太窄？
是动作太大？
```

真实项目不是只收集成功截图，而是要能追溯完整链路。

## 今天必须能说出来

```text
1. 抓取日志必须记录状态、动作、结果。
2. object_x/object_y/hand_x/hand_y/object_type 是 obs。
3. move_dx/move_dy/gripper_width 是 action。
4. success/failure_reason 用来筛选、评估和分析失败。
5. 真实机械臂越早建立日志，后面训练 PyTorch 越容易。
```

