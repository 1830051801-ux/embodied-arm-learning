# Day 14 笔记：把项目讲成面试故事

面试时不要只说：

```text
我用了 YOLO、PyTorch、IK。
```

要讲成完整闭环：

```text
我先做了一个 YOLO + 标定 + IK 的规则抓取系统。
然后记录抓取过程数据。
再用 PyTorch 做行为克隆策略。
最后用规则 baseline 对比测试误差，验证学习策略在多物体场景下更好。
```

## 项目架构怎么讲

```text
camera
-> YOLO detector
-> detection adapter
-> pixel-to-robot calibration
-> PyTorch policy
-> IK
-> dry-run serial command
-> grasp log
-> policy training / evaluation
```

## 你真正展示的能力

```text
1. 视觉感知：YOLO 检测物体类别和 bbox。
2. 坐标转换：像素坐标到机械臂坐标。
3. 策略学习：PyTorch 学习状态到动作。
4. 运动学：IK 把目标动作转成关节角。
5. 工程安全：dry-run、限幅、串口协议确认。
6. 数据闭环：记录抓取日志，再训练和评估。
7. 实验意识：和规则 baseline 做对比。
```

## 和岗位 JD 的对应关系

如果岗位写：

```text
模仿学习 / 强化学习
机器人仿真
Real2Sim2Real
具身数据构建
VLA / VLM
遥操作 / 数据采集
```

你可以对应说：

```text
我目前完成的是一个机械臂具身学习最小闭环：
感知、标定、策略、IK、日志、评估。
下一步会把日志规模扩大，引入仿真和更强策略模型。
```

## 不要夸大的地方

不要说：

```text
我已经训练了 VLA 大模型。
```

可以说：

```text
我做的是 VLA/具身策略前置基础：感知到动作的数据闭环和行为克隆策略。
```

这更真实，也更像工程能力。

