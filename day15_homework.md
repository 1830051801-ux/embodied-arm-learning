# Day 15 小作业 - 闭环 rollout

今天你要搞懂这件事：

```text
单次预测动作不等于完整机械臂控制。
真实机械臂通常需要：观察 -> 动作 -> 状态变化 -> 再观察 -> 再动作。
```

这叫 closed-loop，闭环控制。

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 15
```

重点看：

```text
distance_to_target
```

你会看到类似：

```text
step=00 distance_to_target=0.26449
step=01 distance_to_target=0.17527
step=02 distance_to_target=0.11750
...
step=11 distance_to_target=0.02038
```

这说明策略在连续多步把手爪推向目标。

## open-loop 和 closed-loop

open-loop 开环：

```text
一开始算好动作
后面直接执行
执行过程中不重新观察
```

问题：

```text
机械臂有误差
物体可能动
夹爪可能打滑
坐标可能有偏差
系统无法修正
```

closed-loop 闭环：

```text
每一步重新观察当前状态
根据新状态再决定下一步动作
```

优点：

```text
可以逐步修正误差
```

## rollout 是什么

rollout 就是连续运行策略：

```text
state_0 -> action_0 -> state_1
state_1 -> action_1 -> state_2
state_2 -> action_2 -> state_3
```

你要检查：

```text
distance_to_target 是否越来越小
```

## 动手改一下：换目标点

打开：

```text
day15_step_by_step.py
```

找到：

```python
object_xy = (0.35, -0.20)
hand_xy = [0.00, 0.00]
```

改成：

```python
object_xy = (-0.30, 0.25)
hand_xy = [0.10, -0.10]
```

再跑：

```powershell
.\run_day.bat 15
```

观察 `distance_to_target` 是否仍然下降。

## 动手改一下：减少 rollout 步数

找到：

```python
for step in range(12):
```

改成：

```python
for step in range(4):
```

再跑：

```powershell
.\run_day.bat 15
```

观察最后距离是不是还比较大。

这说明：

```text
有些任务需要多步闭环，不是一两步就能到位。
```

## 和真实机械臂的关系

真实机械臂闭环时，每一步应该更新：

```text
当前 hand_xy
当前关节角
当前目标位置
是否已经接近目标
夹爪是否该闭合
```

这些状态可以来自：

```text
FK 正运动学
编码器
相机重新检测
夹爪/力传感器
```

## 今天必须能说出来

```text
1. 单步预测只输出一次 action。
2. 闭环 rollout 会反复观察、动作、更新状态。
3. distance_to_target 下降说明策略在靠近目标。
4. 真实机械臂需要闭环来修正执行误差。
5. 后面的 ACT/DP/VLA 都是在更复杂地处理动作序列和时序信息。
```

