# 第二周计划：从教学链路走向真实机械臂项目

第一周你学完的是最小闭环：

```text
YOLO -> 标定 -> PyTorch策略 -> IK -> dry-run命令
```

第二周开始把里面的“假数据”逐步换成真实项目需要的模块。

## Day 8: 相机标定

目标：

```text
理解 pixel_x/pixel_y 为什么要变成 robot_x/robot_y。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day8_step_by_step.py
```

学会标准：

```text
你能解释为什么 YOLO bbox 中心点不能直接发给机械臂。
你能解释标定点是什么。
你能看懂标定误差。
```

## Day 9: 真实 YOLO 输出接口

目标：

```text
把真实 YOLO 程序输出成 sample_yolo_detections.json 那样的格式。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day9_step_by_step.py
```

学会标准：

```text
你能解释为什么要有 adapter。
你能把原始 YOLO 输出转换成标准 JSON。
你能指出 class_name/confidence/bbox 分别给后面哪一环使用。
```

## Day 10: 串口 dry-run

目标：

```text
找到 COM 口、确认波特率、确认命令协议，但仍然不直接驱动电机。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day10_step_by_step.py
```

学会标准：

```text
你能解释 dry_run=True 为什么安全。
你能看懂 port/baudrate/newline/command 分别是什么。
你能说出接真机前必须确认哪些协议细节。
```

## Day 11: 真实抓取日志

目标：

```text
把 YOLO、坐标、动作、成功/失败记录成 CSV。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day11_step_by_step.py
```

学会标准：

```text
你能解释为什么日志必须同时记录状态、动作和结果。
你能指出哪些列作为PyTorch输入，哪些列作为训练目标。
你能说明success/failure_reason用于评估和筛选数据。
```

## Day 12: 用真实日志训练策略

目标：

```text
把第一批真实抓取数据喂给 PyTorch 策略。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day12_step_by_step.py
```

学会标准：

```text
你能从CSV里指出哪些列是obs，哪些列是action。
你能解释为什么第一版先筛选success=true样本。
你能理解一两条日志只能演示流程，不能训练出泛化能力。
```

## Day 13: 规则 vs 学习策略对比

目标：

```text
比较 YOLO+规则 和 YOLO+PyTorch 策略的抓取成功率。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day13_step_by_step.py
```

学会标准：

```text
你能解释baseline是什么。
你能用测试误差说明学习策略是否优于规则策略。
你能把这个对比讲成面试里的项目亮点。
```

## Day 14: 项目总结

目标：

```text
整理成面试能讲的具身智能机械臂项目。
```

运行：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day14_step_by_step.py
```

学会标准：

```text
你能用一段话讲清楚项目闭环。
你能诚实区分“已完成的最小闭环”和“未来VLA升级”。
你能把YOLO、PyTorch、IK、日志、baseline对比都讲到同一个项目里。
```
