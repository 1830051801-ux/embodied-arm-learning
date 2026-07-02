# 学习总索引

课程目录：

```text
C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course
```

推荐入口：

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_day.py --list
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_day.py --day 1
```

Windows 批处理入口：

```powershell
.\run_day.bat 1
.\run_day.bat 8
```

## Day 1 到 Day 14

```text
Day 1  PyTorch 最小训练闭环
Day 2  机械臂状态 -> 动作
Day 3  YOLO 输出 -> PyTorch 输入
Day 4  IK：目标位置 -> 关节角
Day 5  抓取数据 CSV -> 模型
Day 6  评估模型有没有学会
Day 7  dry-run 控制命令
Day 8  相机标定 pixel -> robot
Day 9  YOLO 输出 adapter
Day 10 串口 dry-run 模板
Day 11 抓取日志
Day 12 从日志训练策略
Day 13 规则 baseline vs 学习策略
Day 14 项目总结成面试说法
Day 15 闭环策略 rollout
Day 16 动作序列 / Action Chunking
Day 17 观察历史 / Temporal Context
Day 18 Diffusion Policy 去噪直觉
Day 19 语言条件策略 / VLA 最小概念
Day 20 图像输入策略 / Vision to Action
Day 21 多模态融合 / Mini VLA
```

## 检查命令

```powershell
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_week1_check.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_week2_check.py
D:\EmbodiedAI\mujoco-venv\Scripts\python.exe .\run_course_check.py
```

## 学习方法

每一天按这个顺序：

```text
1. 打开 dayN_notes.md
2. 运行 python run_day.py --day N
3. 看输出里的 Step 0, Step 1, Step 2
4. 用自己的话解释当天最后的 Meaning
5. 再改一个小参数重新跑
```
