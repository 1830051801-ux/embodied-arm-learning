# Day 13 小作业 - baseline 怎么证明 PyTorch 有价值

今天你要搞懂这件事：

```text
不能只说“我用了 PyTorch”。
要证明 PyTorch 比简单规则更好，必须有 baseline 对比。
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 13
```

重点看：

```text
rule baseline
learned policy
Per-object movement error
```

你会看到类似：

```text
rule baseline move_error ≈ 0.027
learned policy move_error ≈ 0.0036
```

这说明在这个数据规律下，学习策略整体更接近专家动作。

## 为什么 cup 上规则可能更好

脚本里的规则是：

```text
永远朝物体移动 45%
```

专家数据里 cup 刚好也是：

```text
cup -> 45%
```

所以 cup 上规则误差可能是 0。

但 bottle 和 box 不是 45%：

```text
bottle -> 35%
box    -> 55%
```

所以规则在 bottle/box 上变差。

这说明：

```text
规则在假设刚好成立时很好；
学习策略的价值在于从数据里学到不同情况的不同行为。
```

## 动手改一下：让规则更强

打开：

```text
day13_step_by_step.py
```

找到：

```python
move_xy = 0.45 * (object_xy - hand_xy)
```

这是规则 baseline。

如果你把专家行为也改成全部 0.45：

```python
gain_table = torch.tensor([0.45, 0.45, 0.45])
```

再跑：

```powershell
.\run_day.bat 13
```

你会发现规则可能不比 PyTorch 差。

这说明：

```text
简单问题不一定需要 PyTorch。
```

## 动手改一下：让物体差异更大

把专家行为改成：

```python
gain_table = torch.tensor([0.45, 0.20, 0.70])
```

再跑：

```powershell
.\run_day.bat 13
```

观察：

```text
bottle/box 上 rule error 是否变大？
learned policy 是否还能学到较低误差？
```

这说明：

```text
场景越复杂，数据驱动策略越可能有价值。
```

## 面试时怎么说

不要说：

```text
我用 PyTorch 替代规则。
```

更好的说法：

```text
我先实现 YOLO + 标定 + 规则 + IK 的工程 baseline，
再记录抓取数据训练 PyTorch 策略，
并在未见过的测试样本上按物体类别比较动作误差和成功率。
当场景存在物体差异或遮挡变化时，学习策略优于固定规则。
```

## 和你的机械臂项目的关系

你的第一版可以是：

```text
YOLO -> 标定 -> 手写规则 -> IK -> dry-run
```

然后升级为：

```text
YOLO -> 标定 -> PyTorch 策略 -> IK -> dry-run
```

但升级是否有价值，要看：

```text
测试误差是否更低
真实抓取成功率是否更高
不同物体上的表现是否更稳
```

## 今天必须能说出来

```text
1. baseline 是对照组。
2. YOLO + 规则是合理的第一版，不是错误方案。
3. PyTorch 策略必须用测试指标证明价值。
4. 按物体类别看误差能发现规则在哪些场景失败。
5. 面试时要讲“比较和迭代”，不要只讲“用了 AI”。
```

