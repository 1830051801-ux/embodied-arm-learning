# 第一周验收：你应该已经学会什么

如果你按 Day 1 到 Day 7 学完，应该能解释这条链：

```text
YOLO
-> bbox/class
-> 标定
-> PyTorch 策略
-> IK
-> dry-run 控制命令
```

## 你必须能回答的问题

### 1. PyTorch 是什么

合格回答：

```text
PyTorch 是深度学习库，用来训练模型从数据里学习输入到输出的规律。
在机械臂里，它可以学习“状态 -> 动作”。
```

### 2. YOLO 和 PyTorch 策略有什么区别

合格回答：

```text
YOLO 负责识别物体和检测框。
PyTorch 策略负责根据物体类型、坐标、机械臂状态决定下一步动作。
```

### 3. 为什么 bbox 不能直接给机械臂

合格回答：

```text
bbox 是图像像素坐标，机械臂需要自己坐标系下的 x/y/z。
中间需要相机标定。
```

### 4. IK 是什么

合格回答：

```text
IK 是逆运动学，把目标位置转换成关节角。
PyTorch 可以决定目标动作，IK 把动作变成电机能执行的角度。
```

### 5. 为什么要记录抓取数据

合格回答：

```text
PyTorch 需要很多“状态 -> 正确动作”的示范。
真实机械臂抓取日志可以用来训练和改进策略。
```

### 6. 为什么训练后还要评估

合格回答：

```text
训练 loss 只能说明模型拟合训练数据。
测试误差才能说明模型对没见过的新情况有没有用。
```

### 7. 为什么接真机前要 dry-run

合格回答：

```text
模型或 IK 输出可能超限或方向错误。
dry-run 先打印命令，确认协议、角度范围、夹爪范围和急停，再低速测试。
```

## 你必须能运行的命令

进入课程目录：

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
```

运行第一周检查：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_week1_check.py
```

看到：

```text
WEEK 1 CHECK PASSED
```

说明脚本环境没问题。

## 你必须能改的小练习

### 练习 1

在 `day1_step_by_step.py` 里，把：

```python
y = 2 * x + 1
```

改成：

```python
y = 3 * x - 2
```

重新运行，看模型是否学到新的规律。

### 练习 2

在 `sample_yolo_detections.json` 里，把 bottle 的 bbox 改一下，重新运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day3_step_by_step.py
```

观察 `center_px` 和 `robot object_xy` 是否变化。

### 练习 3

在 `day7_step_by_step.py` 里，把：

```python
q1_deg = -52.39
```

改成：

```python
q1_deg = -140.0
```

观察安全限幅是否生效。

## 第一周结束后你的位置

你还不是在训练大型 VLA 模型，但你已经掌握了一个机械臂具身学习项目的最小闭环：

```text
感知 -> 坐标 -> 策略 -> 运动学 -> 控制命令 -> 数据记录 -> 评估
```

下一周应该进入：

```text
真实 YOLO 摄像头
真实标定
真实机械臂串口 dry-run
更多抓取日志
更接近 ACT/DP/VLA 的策略结构
```

