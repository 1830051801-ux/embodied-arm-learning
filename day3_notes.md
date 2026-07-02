# Day 3 笔记：YOLO 怎么接进 PyTorch

Day 2 里，PyTorch 策略需要这样的输入：

```text
object_x, object_y, hand_x, hand_y, is_cup, is_bottle, is_box
```

Day 3 要回答：

```text
这些 object_x/object_y/object_type 从哪里来？
```

答案：

```text
object_type 来自 YOLO 的 class_name
object_x/object_y 来自 YOLO bbox 中心点 + 相机标定
```

## YOLO 输出什么

一个典型 YOLO 检测结果：

```json
{
  "class_name": "bottle",
  "confidence": 0.91,
  "bbox": [430, 220, 500, 370]
}
```

含义：

```text
class_name：识别出来是什么物体
confidence：置信度
bbox：检测框，格式是 [x1, y1, x2, y2]
```

## bbox 怎么变成中心点

```text
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

例子：

```text
bbox = [430, 220, 500, 370]
center = [465, 295]
```

这还是图片里的像素坐标，不是机械臂坐标。

## 像素坐标为什么不能直接发给机械臂

因为机械臂不懂：

```text
图片第 465 个像素，第 295 个像素
```

机械臂需要的是：

```text
目标在机械臂坐标系下 x=多少米，y=多少米
```

所以需要标定：

```text
pixel_x/pixel_y -> robot_x/robot_y
```

这节课用简单线性映射教学。真实项目里要换成九点标定或单应性矩阵。

## YOLO 和 PyTorch 的关系

YOLO 不是 PyTorch 策略的替代品。

```text
YOLO：告诉你看到了什么、在图像哪里
PyTorch策略：根据物体类型、位置、机械臂状态决定怎么抓
```

完整链路：

```text
camera image
-> YOLO bbox/class
-> bbox center
-> calibration
-> robot state tensor
-> PyTorch policy
-> action
```

## 今天你必须能说出来

```text
1. bbox 是检测框，不是机械臂坐标。
2. bbox 中心点还是像素坐标。
3. 标定把像素坐标变成机械臂坐标。
4. class_name 要变成 one-hot 数字。
5. 这些数字拼起来才是 PyTorch 策略输入。
```

