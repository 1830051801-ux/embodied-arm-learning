# Day 15 笔记：闭环策略 rollout

前面很多脚本只做了一次预测：

```text
状态 -> PyTorch策略 -> 动作
```

但真实机器人通常要循环：

```text
观察当前状态
-> 策略输出动作
-> 执行动作
-> 状态变化
-> 再观察
-> 再输出动作
```

这叫闭环。

## open-loop 是什么

open-loop 开环：

```text
一开始算好一串动作，然后直接执行。
```

问题：

```text
如果物体动了、机械臂有误差、夹爪打滑，系统不会纠正。
```

## closed-loop 是什么

closed-loop 闭环：

```text
每执行一步，都重新看状态，再决定下一步。
```

优点：

```text
可以根据误差逐步修正。
```

## rollout 是什么

rollout 就是把策略连续运行多步：

```text
state_0 -> action_0 -> state_1
state_1 -> action_1 -> state_2
state_2 -> action_2 -> state_3
```

你可以观察：

```text
distance_to_target 是否越来越小
```

## 今天你必须能说出来

```text
1. 一次预测不等于完整机器人控制。
2. 闭环策略会反复观察和修正。
3. rollout 是连续运行策略多步。
4. 如果距离目标越来越小，说明策略在这个任务上有效。
```

