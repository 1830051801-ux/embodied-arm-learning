# Day 5 小作业 - 数据怎么变成 PyTorch 模型

今天你要搞懂这件事：

```text
训练机械臂不是对着 PyTorch 说“帮我抓杯子”。
训练机械臂是给 PyTorch 很多行样本：
当前看到了什么、机械臂在哪、正确动作是什么。
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 5
```

重点看这几行：

```text
obs shape: (12, 7)
action shape: (12, 3)
epoch=300 loss=0.00000000
prediction
correct
```

意思是：

```text
12 行训练样本
每行输入 7 个数字
每行正确动作 3 个数字
模型最后基本学会了这些样本
```

## 打开 CSV 看数据

打开这个文件：

```text
day5_demo_grasp_log.csv
```

每一行都是一次示范：

```text
object_type, object_x, object_y, hand_x, hand_y, move_dx, move_dy, gripper_width
```

含义：

```text
object_type    YOLO 识别到的物体类型
object_x/y     标定后的物体坐标
hand_x/y       当前机械臂末端位置
move_dx/dy     这一步应该怎么移动
gripper_width  夹爪应该张多大
```

## 一行数据怎么变成训练样本

比如一行是：

```text
object_type=box
object_x=0.3357
object_y=0.4378
hand_x=-0.1604
hand_y=-0.2227
move_dx=0.2233
move_dy=0.2972
gripper_width=0.0900
```

PyTorch 输入 `obs` 是：

```text
[0.3357, 0.4378, -0.1604, -0.2227, 0, 0, 1]
```

最后三个 `[0,0,1]` 表示 box。

正确答案 `action` 是：

```text
[0.2233, 0.2972, 0.0900]
```

训练就是让模型学：

```text
obs -> action
```

## 动手改一下

打开：

```text
day5_step_by_step.py
```

找到：

```python
rows = generate_demo_rows()
```

把生成数量改大：

```python
rows = generate_demo_rows(num_rows=100)
```

再跑：

```powershell
.\run_day.bat 5
```

观察：

```text
obs shape
action shape
loss
prediction/correct
```

你应该看到 `obs shape` 从 `(12, 7)` 变成 `(100, 7)`。

## 为什么这和真实机械臂有关

以后你的真实系统要记录的不是假数据，而是真实数据：

```text
摄像头图片路径
YOLO 物体类别
YOLO bbox
标定后的 object_x/object_y
当前关节角
当前手爪位置
执行的动作
是否抓取成功
失败原因
```

数据越真实、越多、覆盖情况越全，PyTorch 才越可能学到稳定策略。

## 今天必须能说出来

```text
1. PyTorch 不会凭空会抓取，它靠数据学习。
2. CSV 每一行就是一条 state -> action 示范。
3. obs 是输入，action 是正确答案。
4. 真实机械臂要记录成功和失败，失败也有学习价值。
5. 没有数据闭环，就只是 YOLO + 规则；有数据闭环，才开始接近具身智能项目。
```

