# Day 4 笔记：为什么需要 IK

Day 3 之后，我们已经能得到：

```text
object_type, object_x, object_y, hand_x, hand_y
```

PyTorch 策略可以输出：

```text
move_dx, move_dy, gripper_width
```

但是电机不能直接执行：

```text
move_dx = 0.12 m
move_dy = -0.04 m
```

电机需要的是：

```text
1 号关节转多少度
2 号关节转多少度
夹爪张多少
```

所以中间需要 IK。

## FK 是什么

FK = Forward Kinematics，正运动学。

```text
已知关节角 -> 算末端位置
```

例子：

```text
q1 = 20 度
q2 = 80 度
-> 算出夹爪在 x/y 哪里
```

## IK 是什么

IK = Inverse Kinematics，逆运动学。

```text
已知目标位置 -> 算关节角
```

例子：

```text
目标 x=0.30, y=0.10
-> 算 q1 应该多少度，q2 应该多少度
```

## PyTorch 和 IK 分工

PyTorch 策略：

```text
根据物体类型、坐标、机械臂状态，决定下一步想去哪里、夹爪张多大。
```

IK：

```text
把“想去哪里”变成每个关节角度。
```

控制板：

```text
把关节角发给电机执行。
```

## 完整链路

```text
YOLO
-> object_type + bbox
-> 标定得到 object_x/object_y
-> PyTorch 输出 move_dx/move_dy/gripper_width
-> 计算 next_hand_x/next_hand_y
-> IK 输出 q1/q2
-> 控制板执行
```

## 今天你必须能说出来

```text
1. FK 是关节角 -> 末端位置。
2. IK 是目标位置 -> 关节角。
3. PyTorch 不一定直接输出电机角度。
4. PyTorch 可以输出目标动作，然后 IK 转成关节角。
5. 真实机械臂必须检查目标是否超出可达范围。
```

