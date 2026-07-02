# Day 9 小作业 - YOLO adapter 和标准 JSON

今天你要搞懂这件事：

```text
真实 YOLO 程序的输出格式不一定固定。
机械臂后面的标定、PyTorch、IK、dry-run 不应该直接依赖某个 YOLO 库的原始字段。
所以中间要有 adapter，把各种 YOLO 输出统一成标准 JSON。
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 9
```

重点看：

```text
raw YOLO-like output
standard detection JSON
chosen target
bbox_center
```

## 原始 YOLO 输出

脚本读取：

```text
day9_raw_yolo_output.json
```

它的字段是：

```json
{
  "width": 640,
  "height": 480,
  "boxes": [
    {
      "label": "bottle",
      "score": 0.91,
      "xyxy": [430, 220, 500, 370]
    }
  ]
}
```

这是假设某个 YOLO 库给出的格式。

## 标准输出

adapter 会生成：

```text
day9_standard_yolo_detections.json
```

标准格式是：

```json
{
  "image_size": [640, 480],
  "detections": [
    {
      "class_name": "bottle",
      "confidence": 0.91,
      "bbox": [430.0, 220.0, 500.0, 370.0]
    }
  ]
}
```

后面的模块只认这个格式。

## 动手改一下：让 cup 变成最高置信度

打开：

```text
day9_raw_yolo_output.json
```

把 cup 的：

```json
"score": 0.86
```

改成：

```json
"score": 0.95
```

再跑：

```powershell
.\run_day.bat 9
```

你应该看到：

```text
chosen target: cup
```

## 动手改一下：加入不支持的类别

在 `boxes` 里加一个：

```json
{
  "label": "phone",
  "score": 0.99,
  "xyxy": [100, 100, 200, 200]
}
```

再跑：

```powershell
.\run_day.bat 9
```

你会发现 `phone` 不会进入标准输出。

原因是脚本只支持：

```python
SUPPORTED_CLASSES = {"cup", "bottle", "box"}
```

这就是工程里的类别白名单。

## 为什么 adapter 很重要

如果没有 adapter，后面的代码会到处写：

```text
label 还是 class_name？
score 还是 confidence？
xyxy 还是 bbox？
width/height 还是 image_size？
```

这样一换 YOLO 模型，整个项目都要改。

有 adapter 后：

```text
换 YOLO 只改 adapter
标定、PyTorch、IK、dry-run 都不用改
```

## 今天必须能说出来

```text
1. YOLO 原始输出不应该直接绑死到控制代码。
2. adapter 负责把原始输出统一成标准 JSON。
3. 后面的 PyTorch 策略只需要 class_name、confidence、bbox。
4. bbox 中心点进入标定，类别进入 one-hot。
5. 换 YOLO 模型时，优先只改 adapter。
```

