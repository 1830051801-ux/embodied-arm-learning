# Day 16 笔记：动作序列 / Action Chunking

Day 15 的策略每次只输出一步：

```text
state -> action
```

例如：

```text
move_dx, move_dy
```

但很多机器人策略会一次输出一小段动作：

```text
state -> action_0, action_1, action_2, ...
```

这叫 action chunking。

## 为什么要输出动作块

单步策略：

```text
每一步都重新预测。
```

优点：

```text
反馈及时。
```

缺点：

```text
动作可能抖动，长动作规划能力弱。
```

动作块策略：

```text
一次预测未来几步动作。
```

优点：

```text
动作更连续，更像一小段计划。
```

缺点：

```text
如果环境变化太快，要及时重新规划。
```

## 和 ACT 的关系

ACT = Action Chunking with Transformers。

它的核心思想之一就是：

```text
根据当前观察，一次预测一段动作序列。
```

我们今天不用 Transformer，先用普通 MLP 学动作块，让你理解 shape 和概念。

## 输出 shape

如果动作维度是：

```text
move_dx, move_dy
```

动作块长度是 3：

```text
[[dx0, dy0],
 [dx1, dy1],
 [dx2, dy2]]
```

模型输出 shape：

```text
batch_size x chunk_size x action_dim
```

## 今天你必须能说出来

```text
1. 单步策略一次输出一个动作。
2. 动作块策略一次输出多步动作。
3. ACT 的 Action Chunking 就是这类思想的高级版本。
4. 动作块可以让机器人动作更连续。
```

