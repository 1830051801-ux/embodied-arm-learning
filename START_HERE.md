# 从这里开始学

这套课程不是让你背 PyTorch API，而是让你学会把 PyTorch 用到你的机械臂项目里。

你要学会的主线只有一条：

```text
YOLO 看到物体
-> 得到物体类型和位置
-> PyTorch 根据数据决定怎么抓
-> IK 把动作变成关节角
-> 控制板执行
-> 记录结果继续训练
```

## 每天怎么学

最简单的运行方式：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_day.py --list
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_day.py --day 1
```

完整索引见：

```text
LEARNING_INDEX.md
```

### Day 1: PyTorch 到底在训练什么

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day1_step_by_step.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_01_train_line.py
```

你要看懂：

```text
x 是输入
y 是正确答案
model 是模型
loss 是模型错了多少
optimizer.step() 是修改模型
```

验收标准：

```text
你能解释为什么 loss 会下降。
你能把 y = 2x + 1 改成 y = 3x - 2，并重新训练成功。
```

### Day 2: 机械臂状态怎么变成动作

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day2_step_by_step.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_02_train_robot_action.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_03_use_robot_policy.py
```

你要看懂：

```text
输入不是一个 x，而是 object_x, object_y, hand_x, hand_y, object_type
输出不是 y，而是 move_dx, move_dy, gripper_width
```

验收标准：

```text
你能说清楚 PyTorch 在这里学的是“状态 -> 动作”。
```

### Day 3: YOLO 怎么接进来

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day3_step_by_step.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_04_fake_yolo_to_policy.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_10_yolo_json_to_robot_command.py
```

你要看懂：

```text
YOLO 输出 class_name, confidence, bbox
bbox 中心点是像素坐标
像素坐标要标定成机械臂坐标
```

验收标准：

```text
你能解释 YOLO 不是控制机械臂，它只提供感知结果。
```

### Day 4: 为什么需要 IK

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day4_step_by_step.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_05_policy_to_ik.py
```

你要看懂：

```text
PyTorch 输出的是想移动多少
机械臂电机需要的是关节角
IK 负责 target_xy -> q1/q2
```

验收标准：

```text
你能解释 IK 和 PyTorch 不是同一个东西。
```

### Day 5: 数据怎么变成模型

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day5_step_by_step.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_06_train_from_csv.py
```

你要看懂：

```text
CSV 里每一行是一条示范
PyTorch 从很多示范里学状态到动作
以后真实机械臂抓取日志可以替换这个 CSV
```

验收标准：

```text
你能说清楚为什么“记录数据”比只写规则更重要。
```

### Day 6: 怎么判断模型有没有学会

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day6_step_by_step.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_08_evaluate_policy.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_09_rule_vs_policy.py
```

你要看懂：

```text
loss 越小通常越好
测试误差比训练误差更重要
规则能起步，但学习策略可以覆盖更复杂差异
```

验收标准：

```text
你能解释 YOLO + 规则 和 YOLO + PyTorch 策略的区别。
```

### Day 7: 接真实机械臂前的安全链路

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day7_step_by_step.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\lesson_07_dry_run_robot_command.py
```

你要看懂：

```text
dry-run 只打印命令，不发给电机
真实电机前必须确认 COM 口、波特率、命令格式、关节限位、急停
```

验收标准：

```text
你能看懂 dry_run_robot_command 里每个数字大概代表什么。
```

## 一键检查课程是否能跑

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_course_check.py
```

它会快速检查关键课程脚本是否能运行。

## 第一周验收

学完 Day 1 到 Day 7 后，打开：

```text
WEEK1_CHECKPOINT.md
```

然后运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_week1_check.py
```

看到 `WEEK 1 CHECK PASSED`，说明第一周脚本环境通过。  
能回答里面的问题，才说明你真的理解了第一周内容。

## 第二周入口

打开：

```text
WEEK2_PLAN.md
```

第二周第一课从相机标定开始：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day8_step_by_step.py
```
