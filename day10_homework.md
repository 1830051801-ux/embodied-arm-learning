# Day 10 小作业 - 串口 dry-run 和控制板协议

今天你要搞懂这件事：

```text
PyTorch/IK 最终会变成一条控制命令。
但控制命令不能随便发给真实机械臂。
必须先确认串口、波特率、换行符、命令格式和急停方式。
```

## 先跑一遍

```powershell
cd "C:\Users\ZhuanZ（无密码）\Documents\Codex\2026-06-30\qu\pytorch_robot_course"
.\run_day.bat 10
```

重点看：

```text
SerialConfig(port='COM3', baudrate=115200, newline='\n', dry_run=True)
DRY_RUN_SEND port=COM3 baudrate=115200: 'HOME\n'
DRY_RUN_SEND port=COM3 baudrate=115200: 'J -52.39 163.53 G 34.4\n'
```

这些输出说明：

```text
没有打开 COM 口
没有发送给电机
只是把将来要发送的命令打印出来
```

## 每个字段是什么意思

```text
port       Windows 里的串口号，比如 COM3
baudrate   串口通信速度，比如 115200
newline    命令末尾的换行符，比如 \n 或 \r\n
dry_run    True 表示只打印，不发送
command    真正给控制板的命令文本
```

## 为什么 newline 重要

有些控制板要求：

```text
J -52.39 163.53 G 34.4\n
```

有些要求：

```text
J -52.39 163.53 G 34.4\r\n
```

有些还要求：

```text
帧头 + 数据 + 校验位 + 帧尾
```

如果换行符或协议不对，控制板可能完全不响应，或者解析错误。

## 动手改一下：改端口和波特率

打开：

```text
day10_step_by_step.py
```

找到：

```python
config = SerialConfig(port="COM3", baudrate=115200, dry_run=True)
```

改成：

```python
config = SerialConfig(port="COM5", baudrate=9600, dry_run=True)
```

再跑：

```powershell
.\run_day.bat 10
```

你应该看到：

```text
DRY_RUN_SEND port=COM5 baudrate=9600
```

这说明命令层会使用你配置的串口参数。

## 动手改一下：改换行符

找到：

```python
newline: str = "\n"
```

改成：

```python
newline: str = "\r\n"
```

再跑：

```powershell
.\run_day.bat 10
```

观察输出里的命令末尾是否变成：

```text
\r\n
```

## 什么时候才能 dry_run=False

必须全部确认后才可以：

```text
1. Windows 设备管理器里确认 COM 口
2. 控制板文档确认 baudrate
3. 控制板文档确认命令格式
4. 确认是否需要 \n、\r\n、校验位
5. 确认每个关节角度范围
6. 确认夹爪单位是 mm、角度还是 PWM
7. 确认急停或断电方法
8. 先低速、小角度、空载测试
```

## 和 PyTorch 的关系

PyTorch 不直接控制电机。

真实链路应该是：

```text
PyTorch 策略 -> action
IK -> q1/q2/...
safety limit -> safe q1/q2/...
serial driver -> 控制板协议命令
真实控制板 -> 电机
```

Day 10 学的是 `serial driver` 这一层。

## 今天必须能说出来

```text
1. dry_run=True 时只打印，不打开串口。
2. port 是 COM 口，baudrate 是通信速度。
3. newline 和协议格式必须和控制板一致。
4. 真实发送前必须确认关节限位、夹爪单位、急停。
5. PyTorch 输出动作，但硬件执行要经过安全和协议层。
```

