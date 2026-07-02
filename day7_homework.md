# Day 7 小作业 - dry-run 命令和安全限制

今天你要搞懂这件事：

```text
模型输出动作后，不能直接发给真实电机。
必须先变成命令字符串，再检查关节范围、夹爪范围、协议格式。
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 7
```

重点看：

```text
dry_run_robot_command: J -52.39 163.53 G 34.4
```

这个命令的教学含义是：

```text
J       joint command
-52.39  q1 关节角
163.53  q2 关节角
G       gripper command
34.4    夹爪宽度，单位 mm
```

注意：这只是课程里的教学协议，不一定等于你的真实控制板协议。

## 为什么必须 dry-run

真实机械臂有风险：

```text
关节角超限
坐标方向反了
夹爪夹太紧
速度太快
撞桌子
串口协议写错
急停没准备好
```

所以正确顺序是：

```text
先打印命令
人检查
确认限制
低速小角度测试
最后才真实发送
```

## 动手改一下：测试安全限制

打开：

```text
day7_step_by_step.py
```

找到：

```python
q1_deg = -52.39
q2_deg = 163.53
gripper_width = 0.0344
```

改成：

```python
q1_deg = -140.0
q2_deg = 190.0
gripper_width = 0.150
```

再跑：

```powershell
.\run_day.bat 7
```

你应该看到输出被限制成：

```text
q1=-90.00
q2=170.00
gripper=100.0 mm
```

这说明安全限制生效了。

## 真实机械臂前必须确认

接真实机械臂之前，你至少要有这些信息：

```text
1. COM 口，比如 COM3
2. 波特率，比如 115200
3. 控制板真实命令协议
4. 每个关节的最小/最大角度
5. 夹爪张开/闭合的范围
6. 是否有速度限制
7. 急停或断电方法
8. 机械臂前方是否清空
```

## 和 PyTorch 的关系

PyTorch 输出的可能是：

```text
move_dx, move_dy, gripper_width
```

IK 输出的是：

```text
q1, q2
```

dry-run 层负责把它们变成：

```text
可检查、可限制、可替换为真实控制板协议的命令
```

## 今天必须能说出来

```text
1. dry-run 是只打印，不发送。
2. 模型输出和 IK 输出都必须做安全检查。
3. 课程里的 J q1 q2 G gripper_mm 是教学格式。
4. 真实机械臂要换成真实控制板协议。
5. 没有急停和限位确认，不应该让模型直接控制电机。
```

