# Day 5 笔记：数据怎么变成模型

前几天我们已经知道：

```text
PyTorch 学的是 状态 -> 动作
```

今天要回答：

```text
这些“正确动作”从哪里来？
```

答案：

```text
从你的机械臂抓取记录来。
```

## 为什么要记录数据

如果你只写规则：

```text
看到瓶子 -> 往目标点移动 -> 夹爪关闭
```

系统只能按你写死的逻辑做。

如果你记录很多次抓取：

```text
当时看到什么
物体在哪里
机械臂在哪里
执行了什么动作
成功还是失败
```

PyTorch 就能从这些记录里学习：

```text
什么状态下应该怎么动更容易成功
```

## CSV 应该记录什么

这节课用的最小 CSV：

```text
object_type, object_x, object_y, hand_x, hand_y, move_dx, move_dy, gripper_width
```

含义：

```text
object_type：YOLO 识别的类别
object_x/object_y：标定后的机械臂坐标
hand_x/hand_y：机械臂当前末端位置
move_dx/move_dy：这一步执行的动作
gripper_width：夹爪张开宽度
```

真实项目可以继续加：

```text
image_path
yolo_confidence
bbox_x1, bbox_y1, bbox_x2, bbox_y2
joint_1, joint_2, joint_3
success
failure_reason
```

## 训练时怎么用 CSV

把 CSV 分成两部分：

输入 obs：

```text
object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box
```

正确答案 action：

```text
move_dx, move_dy, gripper_width
```

然后训练：

```text
obs -> model -> pred_action
pred_action 和 action 比 loss
loss.backward()
optimizer.step()
```

## 今天你必须能说出来

```text
1. PyTorch 训练需要很多“状态 -> 正确动作”的样本。
2. CSV 是最简单的数据集格式。
3. YOLO、标定、机械臂状态、动作都要记录。
4. 真实机械臂项目想变成具身智能项目，必须有数据闭环。
```

