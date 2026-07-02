# Day 9 笔记：真实 YOLO 输出接口

Day 3 用的是假 YOLO JSON：

```json
{
  "image_size": [640, 480],
  "detections": [
    {
      "class_name": "bottle",
      "confidence": 0.91,
      "bbox": [430, 220, 500, 370]
    }
  ]
}
```

Day 9 要解决：

```text
真实 YOLO 程序怎么接到这套链路？
```

答案：

```text
做一个适配器，把真实 YOLO 输出统一成这个 JSON 格式。
```

## 为什么要统一接口

如果后面的 PyTorch/IK 直接依赖某个 YOLO 库的原始输出，换模型就要改很多代码。

更好的方式：

```text
任意 YOLO 输出
-> adapter
-> standard detections JSON
-> PyTorch policy / IK / dry-run command
```

这样后面的机器人链路只认一种格式。

## 标准格式

```json
{
  "image_size": [640, 480],
  "detections": [
    {
      "class_name": "bottle",
      "confidence": 0.91,
      "bbox": [430, 220, 500, 370]
    }
  ]
}
```

字段含义：

```text
image_size：图像宽高
class_name：物体类别
confidence：置信度
bbox：检测框 [x1, y1, x2, y2]
```

## 真实项目里的位置

```text
camera
-> YOLO
-> adapter 输出 standard JSON
-> 标定 pixel->robot
-> PyTorch 策略
-> IK
-> dry-run/控制板
```

## 今天你必须能说出来

```text
1. YOLO 原始输出不应该直接绑死到控制代码。
2. 中间要有标准 JSON 接口。
3. 后面的 PyTorch 策略只需要 class_name 和 bbox 中心点。
4. 换 YOLO 模型时，只改 adapter，后面链路不改。
```

