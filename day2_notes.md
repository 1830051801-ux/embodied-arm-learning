# Day 2 笔记：机械臂状态怎么变成动作

Day 1 学的是：

```text
x -> model -> y
```

Day 2 换成机械臂：

```text
object_x, object_y, hand_x, hand_y, object_type -> model -> move_dx, move_dy, gripper_width
```

## 为什么要这样设计输入

机械臂要抓物体，至少要知道三件事：

```text
1. 物体在哪里：object_x, object_y
2. 夹爪现在在哪里：hand_x, hand_y
3. 物体是什么：cup / bottle / box
```

这些就是状态，也叫 observation。

真实项目里：

```text
object_type 来自 YOLO
object_x/object_y 来自相机标定
hand_x/hand_y 来自机械臂当前状态或正运动学
```

## 为什么输出 move_dx / move_dy / gripper_width

机械臂决策至少要回答：

```text
1. 下一步往 x 方向动多少
2. 下一步往 y 方向动多少
3. 夹爪应该张多大
```

所以输出是：

```text
move_dx, move_dy, gripper_width
```

真实项目里，这个输出后面还要经过：

```text
IK / 控制器 -> 关节角 -> 电机命令
```

## 物体类型为什么要 one-hot

模型不能直接理解字符串：

```text
"cup"
"bottle"
"box"
```

所以要转成数字：

```text
cup    -> [1, 0, 0]
bottle -> [0, 1, 0]
box    -> [0, 0, 1]
```

这叫 one-hot 编码。

## 这节课的训练数据是什么

我们先用一个简单专家规则生成数据：

```text
move_dx = 0.45 * (object_x - hand_x)
move_dy = 0.45 * (object_y - hand_y)
```

夹爪宽度：

```text
cup    -> 0.060 m
bottle -> 0.035 m
box    -> 0.090 m
```

以后真实项目里，这些数据来自你的机械臂日志：

```text
摄像头看到什么
物体坐标是多少
机械臂当时在哪里
你/程序让它怎么动
最后成功还是失败
```

## 今天你必须能说出来

```text
1. PyTorch 输入的是状态，不是图片本身。
2. YOLO 结果会变成状态的一部分。
3. PyTorch 输出的是动作建议。
4. 动作建议还要经过 IK/控制器才能变成电机命令。
5. 训练数据本质是一堆“状态 -> 正确动作”的示范。
```

