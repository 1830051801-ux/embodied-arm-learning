# Day 4 小作业 - IK 为什么夹在 PyTorch 和电机中间

今天你要搞懂这件事：

```text
PyTorch 可以决定“手爪下一步想去哪里”，
但电机真正需要的是“每个关节转多少度”。
```

所以中间需要 IK：

```text
target_x, target_y -> q1, q2
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 4
```

重点看：

```text
target_xy=(0.3000, 0.1000)
IK output q1=-25.28 deg, q2=119.06 deg
FK check_xy=(0.3000, 0.1000)
error=0.00000000 m
```

`error=0` 的意思是：IK 算出来的关节角，用 FK 反算回去，确实到达目标点。

## 你要分清 FK 和 IK

FK：

```text
已知关节角 -> 算手爪位置
q1, q2 -> x, y
```

IK：

```text
已知目标位置 -> 算关节角
x, y -> q1, q2
```

真实机械臂控制里，你经常会这样用：

```text
YOLO/标定得到目标 x,y
PyTorch 决定下一步移动 dx,dy
算出 next_x,next_y
IK 把 next_x,next_y 转成关节角
控制板执行关节角
```

## 动手改一下

打开：

```text
day4_step_by_step.py
```

找到：

```python
target_xy = (0.30, 0.10)
```

改成：

```python
target_xy = (0.40, 0.00)
```

再跑：

```powershell
.\run_day.bat 4
```

观察 `q1/q2` 怎么变化。

## 再试一个超出范围的点

两节机械臂长度是：

```text
link_1 = 0.35 m
link_2 = 0.25 m
max reach = 0.60 m
```

所以这个点够不到：

```python
target_xy = (0.80, 0.00)
```

你应该看到报错：

```text
Target is outside reachable workspace.
```

这就是为什么真实机械臂一定要做安全检查。

## 今天必须能说出来

```text
1. YOLO/标定给的是目标位置。
2. PyTorch 可以输出下一步目标动作。
3. 电机需要关节角，不是 x,y。
4. IK 把目标位置变成关节角。
5. 超出可达范围的目标必须拒绝执行。
```

