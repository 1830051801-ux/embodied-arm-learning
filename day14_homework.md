# Day 14 小作业 - 把项目讲成面试故事

今天你要搞懂这件事：

```text
项目不是技术名词堆叠。
你要能讲清楚：为什么这么做、每个模块负责什么、怎么验证有效、下一步怎么升级。
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 14
```

它会生成：

```text
PROJECT_SUMMARY.md
```

打开读一遍。

## 30 秒版本

你要能这样说：

```text
我做了一个机械臂视觉到动作的最小具身学习闭环。
前端用 YOLO 检测物体和 bbox，通过 adapter 统一检测格式；
再用相机标定把 bbox 中心从像素坐标转换到机械臂坐标；
然后用 PyTorch 策略从 object_xy、hand_xy 和 object_type 预测 move_dx、move_dy、gripper_width；
再通过 IK 转成关节角，并用 dry-run 串口层先打印命令保证安全。
同时我记录抓取日志，把状态、动作、命令和成功失败结果保存成 CSV，
再用成功样本做行为克隆训练，并和规则 baseline 比较测试误差。
```

## 2 分钟版本

按这个结构讲：

```text
1. 问题
机械臂需要从摄像头看到物体，并决定怎么移动和抓取。

2. 第一版工程 baseline
YOLO 检测物体，标定转坐标，规则策略给动作，IK 转关节角，dry-run 输出命令。

3. 为什么加入 PyTorch
规则能跑第一版，但不同物体、位置和失败情况会让 if/else 越来越复杂；
所以记录数据，用 PyTorch 学 state -> action。

4. 数据闭环
每次抓取记录 object_type、bbox、object_xy、hand_xy、action、IK、command、success、failure_reason。

5. 训练方法
第一版用 success=true 的样本做行为克隆，训练策略预测 move_dx、move_dy、gripper_width。

6. 评估方法
和规则 baseline 对比测试误差，并按 cup/bottle/box 分析分类误差。

7. 诚实边界
目前是具身学习最小闭环，不是完整 VLA 大模型；
下一步是接真实相机、真实标定、真实日志、MuJoCo/Isaac Sim 仿真和更强策略模型。
```

## 不要这样说

```text
我训练了 VLA 大模型。
我用 PyTorch 让机械臂自己学会抓东西。
YOLO 不行，所以我换成 PyTorch。
```

这些说法不准确。

## 应该这样说

```text
YOLO 负责感知，PyTorch 负责策略学习，两者不是替代关系。
我先做了可运行的规则 baseline，再用日志数据训练学习策略，并用测试误差验证改进。
目前完成的是从感知到动作的数据闭环和行为克隆基础，为后续 ACT/DP/VLA 做准备。
```

## 和岗位 JD 怎么对应

如果岗位写：

```text
VLA / VLM
模仿学习 / 强化学习
Real2Sim2Real
遥操作 / 数据构建
机器人仿真
动作重定向
```

你可以说：

```text
我当前项目还不是完整 VLA，但已经完成了具身模型训练前最重要的基础：
感知输入标准化、相机标定、state/action 定义、行为克隆训练、日志数据闭环和 baseline 评估。
后续可以把真实抓取日志扩展成模仿学习数据集，并在 MuJoCo/Isaac Sim 中做仿真验证。
```

## 今天必须能说出来

```text
1. 项目主线是 perception -> state -> action -> command -> log -> train -> evaluate。
2. YOLO 和 PyTorch 是分工关系，不是替代关系。
3. 规则 baseline 是第一版工程闭环，PyTorch 是数据驱动升级。
4. 不能夸成完整 VLA，只能说是 VLA/具身策略的基础闭环。
5. 面试要讲证据：测试误差、baseline 对比、日志数据闭环。
```

