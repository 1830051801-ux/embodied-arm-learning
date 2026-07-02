# Day 18 笔记：Diffusion Policy 去噪直觉

Day 16 学的是动作块：

```text
state -> action_chunk
```

Diffusion Policy 也常常输出动作序列，但思路不一样。

它可以理解成：

```text
先从一个很乱的动作序列开始
然后模型一步步去噪
最后得到可执行动作序列
```

## 什么是 clean action chunk

干净动作块：

```text
[[dx0, dy0],
 [dx1, dy1],
 [dx2, dy2]]
```

这是专家动作，机器人应该执行的动作序列。

## 什么是 noisy action chunk

带噪动作块：

```text
clean action chunk + noise
```

例子：

```text
正确动作是向右 0.10
加噪后可能变成向右 0.03 或向左 0.02
```

## 去噪模型学什么

输入：

```text
状态 obs + noisy action chunk
```

输出：

```text
clean action chunk
```

也就是：

```text
模型学习把乱动作修正成好动作
```

## 和真正 Diffusion Policy 的区别

今天是最小概念版：

```text
一次去噪
MLP模型
低维坐标状态
```

真正 Diffusion Policy 更复杂：

```text
多步去噪
噪声时间步 t
图像/机器人状态作为条件
通常用更强的网络结构
```

但核心直觉一样：

```text
条件动作序列去噪
```

## 今天你必须能说出来

```text
1. Diffusion Policy 通常处理动作序列。
2. 它的核心直觉是从噪声动作中去噪。
3. 条件信息是当前观察/机器人状态。
4. Day 16 是直接预测动作块，Day 18 是学习修正带噪动作块。
```

