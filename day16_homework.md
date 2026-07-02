# Day 16 小作业 - Action Chunking

今天你要搞懂这件事：

```text
Day 15 的策略一次输出一步动作。
Day 16 的策略一次输出多步动作序列。
```

这叫 Action Chunking。

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 16
```

重点看：

```text
obs shape: (5000, 4)
action_chunk shape: (5000, 3, 2)
```

含义：

```text
5000  条样本
3     每条样本输出 3 步动作
2     每步动作是 move_dx, move_dy
```

所以模型输出不是：

```text
[move_dx, move_dy]
```

而是：

```text
[[dx0, dy0],
 [dx1, dy1],
 [dx2, dy2]]
```

## 为什么要 action chunk

单步策略：

```text
每一步都重新预测一个动作
```

优点：

```text
反馈及时
```

缺点：

```text
动作可能抖动
缺少一小段连续规划
```

动作块策略：

```text
一次预测未来几步动作
```

优点：

```text
动作更连续
更像一个短期计划
```

缺点：

```text
环境变化太快时，需要及时重新规划
```

## 和 ACT 的关系

ACT = Action Chunking with Transformers。

核心直觉之一就是：

```text
根据当前观察，一次预测一段动作序列。
```

今天我们不用 Transformer，只用 MLP，让你先看懂 shape 和概念。

## 动手改一下：把 chunk 从 3 改成 5

打开：

```text
day16_step_by_step.py
```

找到：

```python
CHUNK_SIZE = 3
```

改成：

```python
CHUNK_SIZE = 5
```

再跑：

```powershell
.\run_day.bat 16
```

你应该看到：

```text
action_chunk shape: (5000, 5, 2)
```

这说明模型一次输出 5 步动作。

## 动手改一下：理解 action_dim

现在：

```python
ACTION_DIM = 2
```

表示每一步动作只有：

```text
move_dx, move_dy
```

如果以后机械臂动作包含夹爪：

```text
move_dx, move_dy, gripper_width
```

那么 action_dim 就会变成 3。

本课程这里先保持 2，不要急着改代码。

## 真实机械臂怎么用 action chunk

一种方式：

```text
预测 5 步动作
执行第 1 步
重新观察
再预测新的 5 步
```

另一种方式：

```text
预测 5 步动作
连续执行前 2-3 步
再重新观察
```

实际怎么选，要看机械臂速度、传感器延迟、任务稳定性。

## 今天必须能说出来

```text
1. single-step policy 一次输出一个动作。
2. action chunk policy 一次输出多步动作。
3. 输出 shape 是 batch x chunk_size x action_dim。
4. ACT 的 Action Chunking 是这个思想的高级版本。
5. 真实机器人可以执行一部分 chunk 后重新观察和规划。
```

