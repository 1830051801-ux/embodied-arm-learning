# 第二周验收

第二周你应该能解释：

```text
真实 YOLO 输出如何接入
相机标定为什么必要
串口 dry-run 为什么安全
抓取日志如何变成 PyTorch 数据集
规则 baseline 如何和学习策略对比
怎么把项目讲成面试故事
```

## 必须能讲的项目版本

```text
我做了一个机械臂具身学习最小闭环：
YOLO 识别物体，标定把像素坐标转成机械臂坐标，
PyTorch 策略根据物体类型、坐标和机械臂状态预测动作，
IK 把动作转成关节角，串口层先 dry-run 保证安全，
同时记录抓取日志，再用日志训练和评估策略。
```

## 必须能指出的不足

```text
当前课程版本仍是教学模拟数据。
真实项目还需要真实摄像头、真实标定、真实机械臂协议、真实抓取日志和成功率评估。
```

## 必须能运行的命令

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\day14_step_by_step.py
```

运行后会生成：

```text
PROJECT_SUMMARY.md
```

第二周完整检查：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_week2_check.py
```

看到：

```text
WEEK 2 CHECK PASSED
```

说明 Day 8 到 Day 14 的项目化脚本都能运行。
