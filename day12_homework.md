# Day 12 小作业 - 从抓取日志训练 PyTorch 策略

今天你要搞懂这件事：

```text
Day 11 的 CSV 日志不是终点。
它要被转换成 PyTorch 训练数据：
obs -> action
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 12
```

重点看：

```text
successful rows: 1
obs shape: (1, 7)
action shape: (1, 3)
saved: day12_policy_from_log.pt
policy prediction
logged action
```

## 为什么只用 success=true

第一版模仿学习通常先模仿成功动作：

```text
success=true 的行 -> 作为正确示范
success=false 的行 -> 先用来分析失败原因
```

如果把失败动作也直接当“正确答案”，模型可能会学会错误行为。

例如失败样本是：

```text
物体在右边，但动作往左走
```

如果你把它当正确动作训练，模型就会学坏。

## CSV 怎么变成 tensor

一行成功日志：

```text
object_type=bottle
object_x=0.2719
object_y=-0.0917
hand_x=0.1234
hand_y=-0.0441
move_dx=0.0668
move_dy=-0.0240
gripper_width=0.0344
```

转换成输入 obs：

```text
[0.2719, -0.0917, 0.1234, -0.0441, 0, 1, 0]
```

其中：

```text
[0, 1, 0] 表示 bottle
```

正确动作 action：

```text
[0.0668, -0.0240, 0.0344]
```

模型学的是：

```text
obs -> action
```

## 为什么一条成功样本不够

今天只有：

```text
successful rows: 1
```

所以模型能做到：

```text
记住这一条
```

但还不能做到：

```text
换个物体也会
换个位置也会
换个角度也会
换个光照也会
真实机械臂误差下也会
```

这叫不能泛化。

真实项目至少要收集：

```text
不同物体
不同位置
不同角度
不同光照
不同初始手爪位置
成功和失败原因
```

## 动手改一下：增加成功样本

打开：

```text
day11_step_by_step.py
```

在 `rows` 里再加一条 `success=true` 的记录，改一下：

```text
object_x/object_y
hand_x/hand_y
move_dx/move_dy
```

然后先跑：

```powershell
.\run_day.bat 11
```

再跑：

```powershell
.\run_day.bat 12
```

观察：

```text
successful rows
obs shape
action shape
```

如果你加了一条成功样本，应该看到：

```text
successful rows: 2
obs shape: (2, 7)
action shape: (2, 3)
```

## 失败样本以后怎么用

第一阶段：

```text
失败样本用于分析，不直接当正确动作训练
```

后面更高级可以用于：

```text
过滤低质量数据
训练成功率预测器
训练避错策略
强化学习 reward
偏好学习
```

但第一版不要复杂化，先把成功模仿学习跑通。

## 今天必须能说出来

```text
1. 日志 CSV 可以变成 PyTorch 数据集。
2. 第一版模仿学习先用 success=true 样本。
3. object_type 要转 one-hot。
4. obs 是状态，action 是正确动作。
5. 一条成功样本只能演示流程，不能训练出泛化能力。
```

