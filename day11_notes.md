# Day 11 笔记：真实抓取日志

如果你想让 PyTorch 从你的机械臂项目里学习，必须记录数据。

不是只记录：

```text
抓成功了 / 抓失败了
```

而是要记录完整链路：

```text
YOLO 看到了什么
标定后的坐标是多少
机械臂当时在哪里
策略输出了什么动作
IK 算出什么角度
准备发什么命令
最后是否成功
```

## 为什么要记录这么多字段

因为训练时要知道：

```text
输入状态是什么
正确动作是什么
结果好不好
```

如果只记录成功/失败，不记录当时状态，模型没法学习。

## 最小字段

```text
episode_id
step
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

## 和 PyTorch 的关系

训练输入：

```text
object_x, object_y, hand_x, hand_y, object_type
```

训练目标：

```text
move_dx, move_dy, gripper_width
```

评估字段：

```text
success, failure_reason
```

## 今天你必须能说出来

```text
1. 没有日志就没有自己的训练数据。
2. 日志要记录状态、动作和结果。
3. YOLO/标定/策略/IK/命令都应该能追溯。
4. 以后训练策略时，CSV 就是数据集。
```

