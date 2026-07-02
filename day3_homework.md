# Day 3 小作业 - 把 YOLO 输出变成 PyTorch 输入

今天的目标不是训练模型，而是看懂这件事：

```text
YOLO 检测结果不能直接喂给机械臂。
必须先变成 PyTorch 策略能读懂的数字向量。
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 3
```

你要重点看这几行：

```text
chosen target
center_px
robot object_xy
one_hot
obs tensor shape
obs tensor
```

## 你要理解的转换

原始 YOLO 结果：

```text
class_name = bottle
confidence = 0.91
bbox = [430, 220, 500, 370]
```

先算检测框中心：

```text
center_x = (430 + 500) / 2 = 465
center_y = (220 + 370) / 2 = 295
```

再做相机标定，把像素点变成机械臂坐标：

```text
center_px = [465, 295]
robot_xy ≈ [0.2719, -0.0917]
```

再把类别变成 one-hot：

```text
cup    = [1, 0, 0]
bottle = [0, 1, 0]
box    = [0, 0, 1]
```

最后拼成 PyTorch 输入：

```text
[object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box]
```

这就是 Day 2 里策略模型需要的 `state`。

## 动手改一下

打开：

```text
sample_yolo_detections.json
```

把 bottle 的分数从：

```json
"confidence": 0.91
```

改成比 cup 低，比如：

```json
"confidence": 0.50
```

然后再跑：

```powershell
.\run_day.bat 3
```

你应该看到 `chosen target` 变成 cup。

## 为什么这个练习重要

因为真实项目里，摄像头每一帧都会输出很多检测框。

你必须决定：

```text
选哪个物体？
它的中心点在哪里？
这个点对应机械臂坐标是多少？
类别怎么变成模型能读的数字？
```

这些都准备好了，PyTorch 策略才有资格输出动作。

