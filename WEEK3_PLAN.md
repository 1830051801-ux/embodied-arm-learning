# 第三周计划：从一次动作预测到闭环策略

前两周你已经学会：

```text
YOLO -> 标定 -> PyTorch策略 -> IK -> dry-run命令 -> 日志 -> 训练 -> 评估
```

第三周开始学更像机器人策略的概念：

```text
不是只预测一次动作，而是反复观察状态、预测动作、执行、再观察。
```

## Day 15: 闭环 rollout

目标：

```text
理解 PyTorch 策略如何一步步让机械臂末端靠近目标。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day15_step_by_step.py
```

学会标准：

```text
你能解释 open-loop 和 closed-loop 的区别。
你能看懂每一步 distance_to_target 为什么下降。
你能解释为什么真实机器人需要反复观察和纠正。
```

## Day 16: 动作序列 / Action Chunking

目标：

```text
理解为什么有些机器人策略一次输出多步动作，而不是只输出一步。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day16_step_by_step.py
```

学会标准：

```text
你能解释 single-step policy 和 action chunk policy 的区别。
你能看懂模型输出 shape 为什么是 batch x chunk x action_dim。
你能把 action chunking 和 ACT 联系起来。
```

## Day 17: 观察历史 / Temporal Context

目标：

```text
理解为什么机器人策略常常需要最近几帧观察，而不是只看当前一帧。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day17_step_by_step.py
```

学会标准：

```text
你能解释为什么历史状态可以推断目标速度。
你能看懂 current-only policy 和 history policy 的误差对比。
你能把 temporal context 和 ACT/DP/VLA 的时序建模联系起来。
```

## Day 18: Diffusion Policy 去噪直觉

目标：

```text
理解 Diffusion Policy 为什么可以看成“从噪声动作序列逐步去噪成可执行动作序列”。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day18_step_by_step.py
```

学会标准：

```text
你能解释 action chunk、noise、denoise 分别是什么。
你能看懂 noisy_action_chunk 和 clean_action_chunk 的误差。
你能说明 Diffusion Policy 和 Day 16 action chunking 的关系。
```

## Day 19: 语言条件策略 / VLA 最小概念

目标：

```text
理解语言指令如何作为策略输入的一部分，影响机械臂动作。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day19_step_by_step.py
```

学会标准：

```text
你能解释为什么同一个物体位置，在不同语言指令下应该输出不同动作。
你能看懂 language one-hot 如何进入 PyTorch 模型。
你能把这个最小例子和 VLA 的 vision-language-action 联系起来。
```

## Day 20: 图像输入策略 / Vision to Action

目标：

```text
理解图像如何进入 PyTorch 策略模型，而不是先手动变成坐标。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day20_step_by_step.py
```

学会标准：

```text
你能解释 CNN 在这里负责从图像提特征。
你能看懂 image tensor shape。
你能把这个最小例子和 VLA 的 Vision 部分联系起来。
```

## Day 21: 多模态融合 / Mini VLA

目标：

```text
把图像、语言、机器人状态合在一起输入策略模型，形成最小 VLA 结构。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day21_step_by_step.py
```

学会标准：

```text
你能解释 image、language、robot_state 分别提供什么信息。
你能看懂多模态特征如何 concat。
你能说明 mini VLA 和真正 VLA 的区别。
```
