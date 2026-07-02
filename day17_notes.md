# Day 17 笔记：观察历史 / Temporal Context

很多机器人策略不只看当前一帧。

它会看：

```text
最近几帧图像
最近几帧关节状态
最近几步动作
```

这叫 temporal context，时序上下文。

## 为什么当前一帧不够

如果物体是静止的：

```text
当前物体位置就够了。
```

但如果物体在移动：

```text
只看当前位置，不知道它往哪边动。
```

看两帧：

```text
上一帧位置 -> 当前帧位置
```

就能估计速度：

```text
velocity = current - previous
```

## 和机器人有什么关系

抓动态物体、处理滑动、夹爪接触、手臂延迟，都需要时序信息。

例子：

```text
杯子正在向右移动
只看当前帧：伸手到当前位置
看历史帧：预测它下一刻会更靠右
```

## 和 ACT / DP / VLA 的关系

这些模型通常会处理序列：

```text
observation history -> action / action chunk
```

区别只是模型更复杂：

```text
MLP -> Transformer / Diffusion / VLA
```

但核心问题一样：

```text
利用历史信息做更好的动作决策。
```

## 今天你必须能说出来

```text
1. 当前一帧只能告诉位置。
2. 历史帧能推断速度和趋势。
3. 机器人策略常用 observation history。
4. ACT/DP/VLA 都涉及时序建模。
```

