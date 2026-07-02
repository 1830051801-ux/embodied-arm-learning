# Day 6 小作业 - 怎么判断模型有没有真的学会

今天你要搞懂这件事：

```text
脚本跑完不等于模型会了。
训练 loss 下降也不一定等于真实机械臂能用。
必须看测试误差、分类误差、和规则 baseline 的对比。
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 6
```

重点看这些行：

```text
train_loss
test_loss
simple_rule mse
pytorch mse
cup/bottle/box move_error
```

## 今天这次结果怎么读

你会看到类似：

```text
simple_rule    mse=0.0010
pytorch        mse=0.000014
```

这说明 PyTorch 的总体动作误差更小。

再看分类：

```text
cup    move_error simple_rule=0.00000 pytorch=0.00360
bottle move_error simple_rule=0.03775 pytorch=0.00314
box    move_error simple_rule=0.03985 pytorch=0.00344
```

意思是：

```text
规则刚好适合 cup
但 bottle 和 box 的动作规律不一样，规则误差变大
PyTorch 从数据里学到了不同物体的不同行为
```

这就是为什么你的 YOLO + 规则能做第一版，但不能代表完整具身智能。

## 训练集和测试集的区别

训练集：

```text
模型见过的数据
```

测试集：

```text
模型没见过的数据
```

如果：

```text
train_loss 很低
test_loss 很高
```

说明模型可能只是在背训练数据，没有学会真正规律。

真实机械臂里，这种情况很危险：实验室固定位置能抓，换个位置就失败。

## 动手改一下：减少训练轮数

打开：

```text
day6_step_by_step.py
```

找到：

```python
for epoch in range(701):
```

改成：

```python
for epoch in range(51):
```

再跑：

```powershell
.\run_day.bat 6
```

观察：

```text
test_loss 是否变大？
pytorch 的 move_error 是否变大？
```

这会让你看到：训练不够，模型还没学好。

## 再动手改一下：让规则更吃亏

找到：

```python
move_gain = torch.tensor([0.45, 0.35, 0.55])
```

这是不同物体的动作比例：

```text
cup    0.45
bottle 0.35
box    0.55
```

你可以改成：

```python
move_gain = torch.tensor([0.45, 0.20, 0.70])
```

再跑：

```powershell
.\run_day.bat 6
```

观察规则 baseline 在 bottle/box 上是不是更差。

## 和真实机械臂的关系

你的真实项目以后也要这样评估：

```text
训练数据上的误差
新位置上的误差
不同物体类别的误差
不同光照/角度/距离下的误差
规则 baseline 和 PyTorch 策略对比
真实抓取成功率
```

不要只说：

```text
我训练了一个模型
```

要能说：

```text
我用未见过的测试样本评估，按物体类别统计误差，并和规则 baseline 对比。
```

这才像应聘岗位里说的“模型迭代和训练策略优化”。

## 今天必须能说出来

```text
1. train_loss 看模型是否拟合训练数据。
2. test_loss 看模型是否能泛化到新数据。
3. baseline 是比较对象，没有 baseline 就不知道模型值不值得用。
4. 按物体类别看误差，能发现某类物体抓不好。
5. 真实机械臂最终还要看抓取成功率，不只看 loss。
```

