# Day 1 笔记：PyTorch 训练到底在干什么

今天只学一件事：

```text
模型一开始是错的，PyTorch 根据 loss 一点点修改模型参数，让它越来越接近正确答案。
```

## 为什么先学 y = 2x + 1

因为机械臂训练看起来复杂，但形式一样。

最小例子：

```text
输入 x -> 模型 -> 输出 y
```

机械臂：

```text
输入 物体坐标/物体类型/机械臂状态 -> 模型 -> 输出 动作
```

所以先用最简单的 `y = 2x + 1`，把训练过程看懂。

## 五个核心词

### 1. tensor

PyTorch 里的数字表。

```text
x = [[-5], [-4.9], ..., [5]]
```

以后机械臂输入也会变成 tensor：

```text
[object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box]
```

### 2. model

模型就是一个函数。

今天的模型：

```text
y_pred = w * x + b
```

一开始 `w` 和 `b` 是随机的，所以预测不准。

### 3. loss

loss 是“模型错了多少”。

```text
loss 大：模型很错
loss 小：模型接近正确答案
```

### 4. backward

`loss.backward()` 会计算：

```text
w 应该往哪个方向改
b 应该往哪个方向改
```

### 5. optimizer.step

`optimizer.step()` 真正修改模型参数。

```text
改 w
改 b
让下次预测更准
```

## 你今天必须能说出来

```text
1. x 是输入。
2. y 是正确答案。
3. model 用 x 预测 y_pred。
4. loss 比较 y_pred 和 y。
5. backward 计算怎么改。
6. optimizer.step 修改模型。
7. 重复很多次，loss 下降，模型学会规律。
```

## 和你的机械臂项目的关系

今天：

```text
x -> y
```

以后：

```text
物体类型 + 物体坐标 + 当前机械臂状态 -> 下一步动作
```

所以 PyTorch 学的不是“魔法”，而是：

```text
输入状态和正确动作之间的规律。
```

