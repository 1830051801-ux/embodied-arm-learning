# Day 21 笔记：多模态融合 / Mini VLA

VLA = Vision-Language-Action。

最小形式：

```text
image + language + robot_state -> action
```

Day 19 学了：

```text
language + state -> action
```

Day 20 学了：

```text
image + state -> action
```

Day 21 合起来：

```text
image + language + state -> action
```

## 三种输入分别负责什么

### image

告诉模型：

```text
物体在哪里
场景长什么样
```

### language

告诉模型：

```text
任务是什么
比如 pick / push / pull
```

### robot_state

告诉模型：

```text
机械臂现在在哪里
夹爪现在是什么状态
```

## 融合怎么做

最小版本：

```text
CNN(image) -> vision_feature
one_hot(language) -> language_feature
hand_xy -> robot_state_feature

concat(vision_feature, language_feature, robot_state_feature)
-> MLP
-> action
```

## 真正 VLA 更复杂在哪里

今天：

```text
16x16小图 + one-hot语言 + hand_xy -> action
```

真正 VLA：

```text
真实图像/视频 + 大语言模型文本编码 + 机器人状态 -> 动作序列
```

但核心结构一样：

```text
多模态输入 -> 动作输出
```

## 今天你必须能说出来

```text
1. Vision 提供场景信息。
2. Language 提供任务意图。
3. Robot state 提供当前机器人状态。
4. VLA 策略要把这些信息融合后输出动作。
```

