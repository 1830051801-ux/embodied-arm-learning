# Day 12 笔记：用真实日志训练策略

Day 11 生成了抓取日志：

```text
day11_grasp_episode_log.csv
```

Day 12 要做：

```text
读取日志 -> 取出训练输入和正确动作 -> 训练 PyTorch 策略
```

## 为什么先筛选成功样本

模仿学习通常先模仿好的动作。

如果失败动作也直接当成正确答案，模型会学到错误行为。

所以第一版可以：

```text
success=true 的行 -> 用来训练
success=false 的行 -> 用来分析失败原因
```

后面更高级的方法可以利用失败样本做强化学习或偏好学习，但第一步先别复杂化。

## 日志怎么变成训练数据

CSV 里有很多列。

输入 obs：

```text
object_x, object_y, hand_x, hand_y, object_type
```

目标 action：

```text
move_dx, move_dy, gripper_width
```

类别 object_type 要 one-hot：

```text
cup    -> [1, 0, 0]
bottle -> [0, 1, 0]
box    -> [0, 0, 1]
```

最后输入 tensor：

```text
[object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box]
```

## 训练完成后保存什么

保存模型：

```text
day12_policy_from_log.pt
```

以后推理时：

```text
YOLO + 标定 + 当前手位置 -> obs tensor -> policy -> action
```

## 今天你必须能说出来

```text
1. 抓取日志 CSV 可以变成 PyTorch 数据集。
2. 第一版模仿学习通常先用成功样本。
3. object_type 要转 one-hot。
4. 模型输出 move_dx/move_dy/gripper_width。
5. 日志越真实、越多样，策略才越有价值。
```

