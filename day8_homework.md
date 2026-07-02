# Day 8 小作业 - 相机标定 pixel -> robot

今天你要搞懂这件事：

```text
YOLO 给的是图片像素坐标。
机械臂要的是桌面/机械臂坐标。
相机标定负责把 pixel_x,pixel_y 变成 robot_x,robot_y。
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 8
```

重点看：

```text
center_px=(465.0, 295.0)
pred_robot_xy=(0.2746, -0.0914)
true_robot_xy=(0.2719, -0.0917)
calibration_error=0.002786 m
```

`0.002786 m` 约等于 `2.8 mm`。

这说明这个教学标定模型在这个点上的误差大约是 2.8 毫米。

## 为什么这个误差重要

如果标定误差是 3 mm，机械臂可能还能抓到杯子。

如果标定误差是 30 mm，YOLO 就算识别对了，机械臂也会抓偏。

所以真实项目里要关心：

```text
YOLO 检测是否准
标定误差是否小
机械臂执行是否准
```

不要把所有失败都怪 YOLO 或 PyTorch。

## 标定点是什么

脚本里有 9 个点：

```text
pixel=(80,80)    -> robot=(-0.45,  0.2667)
pixel=(320,240)  -> robot=( 0.00,  0.0000)
pixel=(560,400)  -> robot=( 0.45, -0.2667)
```

真实项目里，这些点应该来自你实际测量：

```text
把一个标记点放在桌面某个真实坐标
摄像头里读出它的像素坐标
记录 pixel_x,pixel_y 和 robot_x,robot_y
```

## 动手改一下：换一个 YOLO 检测框

打开：

```text
day8_step_by_step.py
```

找到：

```python
bbox = [430, 220, 500, 370]
```

改成：

```python
bbox = [250, 170, 330, 300]
```

再跑：

```powershell
.\run_day.bat 8
```

观察：

```text
center_px
pred_robot_xy
true_robot_xy
calibration_error
```

你会看到不同像素点会变成不同机械臂坐标。

## 动手改一下：减少标定点

脚本里 `pixel_xy = torch.tensor([...])` 有 9 个标定点。

你可以临时删掉几行，只留下 4 个角点，再跑：

```powershell
.\run_day.bat 8
```

观察 `calibration_error` 是否变大或不稳定。

这能帮助你理解：

```text
标定点越少，覆盖越差，模型越容易在某些区域不准。
```

## 工程提醒

真实工程里，平面标定通常优先用 OpenCV 的单应性矩阵。

这里用 PyTorch 是为了让你直观看懂：

```text
输入 pixel_x,pixel_y
正确答案 robot_x,robot_y
模型通过 loss 学映射
```

这个思想和后面训练机械臂策略是一样的。

## 今天必须能说出来

```text
1. YOLO bbox 中心点是像素坐标，不是机械臂坐标。
2. 标定点是一组 pixel -> robot 的配对数据。
3. 标定误差越小，机械臂目标位置越准。
4. 如果标定错了，YOLO 和 PyTorch 都可能看起来“没用”。
5. 真实项目要记录并验证标定误差。
```

