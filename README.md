# Embodied Arm Learning: PyTorch Mechanical Arm Pipeline

面向视觉机械臂项目的具身学习训练仓库。项目把 YOLO 检测、相机/桌面标定、PyTorch 行为克隆策略、IK、dry-run 机械臂命令、抓取日志和规则基线评估串成一个可运行的最小闭环。

## 项目亮点

- 构建 `camera -> YOLO -> calibration -> PyTorch policy -> IK -> dry-run command -> log -> train/evaluate` 的完整链路。
- 使用 PyTorch 训练 Behavior Cloning 策略，学习 `observation -> action`，用于提升机械臂抓取决策。
- 提供规则基线对比、抓取日志样例、action chunking、闭环 rollout、YOLO JSON 适配等可复现实验。
- 为真实机械臂项目预留 Real2Sim2Real 路线：先用日志和仿真验证策略，再逐步接入实机抓取数据。

## 核心结果

```text
训练样本: 5000
测试样本: 1500
PyTorch policy test MSE: 0.00001443
rule baseline MSE:       0.00102663
标定示例误差:             0.0028 m
闭环 rollout:             0.26449 -> 0.02038
```

这套小课程的目标不是先做炫酷仿真，而是让你从 0 搞懂：

1. PyTorch 到底在训练什么。
2. 为什么机械臂项目需要“数据 -> 模型 -> 动作”。
3. YOLO、坐标、物体类型、IK、PyTorch 分别在系统里负责什么。

## 先记住整体路线

```text
摄像头
-> YOLO 识别物体类型和像素位置
-> 标定得到机械臂坐标
-> PyTorch 策略模型决定下一步动作
-> IK / 控制器把动作变成电机命令
-> 机械臂执行
```

## 课程顺序

### Lesson 01: 最小 PyTorch 训练

文件：

```text
lesson_01_train_line.py
```

目的：让模型学习 `y = 2x + 1`。

为什么做这个：机械臂训练也是输入到输出，只是输入从 `x` 变成了物体坐标、物体类型、机械臂状态。

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_01_train_line.py
```

### Lesson 02: 机械臂动作模仿学习

文件：

```text
lesson_02_train_robot_action.py
```

目的：训练一个小模型，输入物体坐标、夹爪当前位置、物体类型，输出机械臂下一步动作。

为什么做这个：这就是“用 PyTorch 提高抓取决策”的最小版本。

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_02_train_robot_action.py
```

### Lesson 03: 使用训练好的策略

文件：

```text
lesson_03_use_robot_policy.py
```

目的：像真实机械臂一样，给模型一个物体位置和类型，让它输出动作。

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_03_use_robot_policy.py
```

### Lesson 04: YOLO 输出接入 PyTorch 策略

文件：

```text
lesson_04_fake_yolo_to_policy.py
```

目的：把 YOLO 的检测框中心点变成机械臂坐标，再交给 PyTorch 策略输出动作。

为什么做这个：你的真实项目不是手填坐标，而是摄像头和 YOLO 给出物体类型、像素位置，然后机器人系统继续决策。

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_04_fake_yolo_to_policy.py
```

### Lesson 05: PyTorch 策略输出转 IK 关节角

文件：

```text
lesson_05_policy_to_ik.py
```

目的：把 PyTorch 输出的 `move_dx / move_dy` 变成二维机械臂的关节角。

为什么做这个：模型输出的是“想往哪里动”，真实机械臂电机需要的是“每个关节转多少度”。

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_05_policy_to_ik.py
```

### Lesson 06: 从抓取记录训练策略

文件：

```text
lesson_06_train_from_csv.py
```

目的：先生成一份像真实机械臂日志一样的 CSV，再用这份 CSV 训练 PyTorch 策略。

为什么做这个：PyTorch 不是凭空会抓，它要从很多条“当时看到什么、机械臂在哪里、正确动作是什么”的记录里学。

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_06_train_from_csv.py
```

### Lesson 07: dry-run 机械臂命令链路

文件：

```text
lesson_07_dry_run_robot_command.py
```

目的：把前面的 YOLO、PyTorch、IK 串起来，最后生成准备发给控制板的命令，但只打印，不真的发给电机。

为什么做这个：真实机械臂不能一上来就让 AI 控制，必须先验证命令格式、关节角、安全限位。

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_07_dry_run_robot_command.py
```

### Lesson 08: 评估模型到底学会没有

文件：

```text
lesson_08_evaluate_policy.py
```

目的：检查训练出的策略模型在测试数据上的平均误差、最大误差，以及不同物体的夹爪宽度是否合理。

为什么做这个：训练不是看到 loss 变小就完事。你要知道模型有没有在新样本上也能做对。

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_08_evaluate_policy.py
```

### Lesson 09: 硬编码规则 vs PyTorch 策略

文件：

```text
lesson_09_rule_vs_policy.py
```

目的：比较普通规则控制和 PyTorch 策略在同一批测试数据上的误差。

为什么做这个：YOLO + 规则当然可以先做起来，但规则通常很难覆盖不同物体、不同位置、不同抓法。PyTorch 的价值是从数据里学这些差异。

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_09_rule_vs_policy.py
```

### Lesson 10: 从 YOLO 检测结果文件接入

文件：

```text
lesson_10_yolo_json_to_robot_command.py
sample_yolo_detections.json
```

目的：从 JSON 文件读取 YOLO 检测结果，再走 PyTorch 策略、IK、dry-run 命令。

为什么做这个：真实 YOLO 程序通常会输出类别、置信度、bbox。把检测结果变成文件接口，后面的策略和控制链路就可以复用。

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_10_yolo_json_to_robot_command.py
```

## 对应你的真实项目

当前课程里的输入：

```text
object_x, object_y, hand_x, hand_y, object_type
```

以后真实项目里的输入：

```text
YOLO类别, YOLO框中心, 标定后的x/y/z, 当前关节角度, 夹爪状态, 成功/失败记录
```

当前课程里的输出：

```text
move_dx, move_dy, gripper_width
```

以后真实项目里的输出：

```text
关节角度变化, 末端目标姿态, 夹爪开合, 抓取策略
```

## 现在不要混淆的三件事

```text
YOLO：负责看见物体，输出类别和检测框
PyTorch策略：负责根据类别、坐标、当前状态决定怎么动
IK/控制器：负责把动作变成关节角度和电机命令
```

## 接真实机械臂前必须确认

```text
1. 控制板串口号，例如 COM3
2. 波特率，例如 115200
3. 命令格式，例如 J q1 q2 q3 或 G OPEN
4. 每个关节的安全角度范围
5. 急停/断电方式
6. 先 dry-run 打印命令，再低速小角度测试
```

## 为什么一定要记录数据

训练机械臂不是只写规则，而是收集这样的表：

```text
object_type, object_x, object_y, hand_x, hand_y, move_dx, move_dy, gripper_width
```

每一行就是一次“示范”。PyTorch 根据这些示范学习：

```text
看到什么状态 -> 应该输出什么动作
```

## 从 0 学习顺序

```text
1. 先跑 Lesson 01，看懂 loss 为什么下降。
2. 再跑 Lesson 02，看懂机械臂状态如何变成动作。
3. 再跑 Lesson 03，看懂模型保存和加载。
4. 再跑 Lesson 04，看懂 YOLO 输出如何进入策略。
5. 再跑 Lesson 05，看懂 IK 为什么需要。
6. 再跑 Lesson 06，看懂真实数据如何训练模型。
7. 再跑 Lesson 07，看懂发给控制板前的 dry-run。
8. 最后跑 Lesson 08，看懂怎么评估模型有没有学会。
9. 跑 Lesson 09，看懂规则控制和学习策略的区别。
10. 跑 Lesson 10，看懂真实 YOLO 输出怎么接入整条机器人链路。
```
