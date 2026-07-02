# Day 17 作业：让机械臂学会看历史

今天学的是 temporal context：策略不只看当前一帧，还看上一帧/最近几帧。

## 为什么做这个

如果物体不动：

```text
当前坐标 -> 机械臂移动过去
```

就够了。

但如果物体在动，只看当前坐标会慢半拍。看历史帧后，PyTorch 策略可以估计：

```text
上一帧位置 -> 当前帧位置 -> 运动方向/速度 -> 下一步应该抓哪里
```

这就是以后学 ACT、Diffusion Policy、VLA 的基础。

## 怎么运行

在 PowerShell 里运行：

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_day.py --day 17
```

也可以运行：

```powershell
.\run_day.bat 17
```

## 你要看懂的输出

重点看这里：

```text
current_only   mse=...
history        mse=...
```

`current_only` 是只看当前一帧。

`history` 是看上一帧和当前帧。

误差越小，说明动作预测越准。

## 你今天要改的地方

打开：

```text
day17_step_by_step.py
```

找到：

```python
velocity = torch.empty(num_samples, 2).uniform_(-0.06, 0.06)
```

把 `0.06` 改成 `0.12`，再运行一次。

问题：

```text
物体移动更快以后，只看当前一帧的策略是不是更吃亏？
```

## 和你的机械臂项目的关系

你的完整路线是：

```text
YOLO -> 标定 -> PyTorch策略 -> IK -> dry-run命令 -> 日志 -> 训练 -> 评估
```

Day 17 加进去的位置是：

```text
YOLO连续检测多帧 -> 历史观测 -> PyTorch策略 -> IK -> 机械臂动作
```

以后你的机械臂不是只接收：

```text
当前物体坐标
```

而是接收：

```text
上一帧物体坐标 + 当前物体坐标 + 当前手的位置
```

这样它才能对动态目标、滑动物体、延迟、抓取修正做更好的决策。

