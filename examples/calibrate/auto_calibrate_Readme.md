# Auto Calibrate 使用说明

本文档说明如何使用自动标定命令行脚本和 PyQt 图形界面对 SO 系列设备执行自动标定。

当前支持两类设备：

- `tele`: leader / teleoperator
- `robot`: follower / robot

相关文件：

- [auto_calibrate.py](/d:/workspace/lerobot_xlerobot/src/lerobot/motors/auto_calibrate.py)
- [auto_calibrate_ui.py](/d:/workspace/lerobot_xlerobot/src/lerobot/scripts/auto_calibrate_ui.py)
- [auto_calibrate_example.py](/d:/workspace/lerobot_xlerobot/examples/calibrate/auto_calibrate_example.py)

## 功能概览

自动标定流程会完成以下工作：

- 连接设备
- 逐个关节探索机械边界
- 计算 `homing_offset`、`range_min`、`range_max`
- 生成 `MotorCalibration`
- 保存 calibration JSON 文件

图形界面额外提供：

- 串口自动检测
- Linux 下 `/dev/ttyACM*` 过滤
- Linux 串口授权按钮
- 机械臂检测按钮
- 标定暂停 / 继续
- 日志实时输出
- 输出路径预览

## 命令行用法

### 必选参数

- `--port`: 串口号，例如 `COM9` 或 `/dev/ttyACM0`
- `--device-type`: 设备类型，只能是 `tele` 或 `robot`

### 可选参数

- `--id`: 设备 ID，默认 `my_so101`
- `--try-torque`
- `--max-torque`
- `--torque-step`
- `--explore-velocity`
- `--wait-time-s`
- `--velocity-threshold`
- `--position-tolerance`
- `--log`: 是否输出终端日志，例如 `--log=false`
- `--yes`: 跳过启动前确认

### 示例

标定 tele：

```bash
python examples/calibrate/auto_calibrate_example.py --port COM9 --device-type tele
```

标定 robot：

```bash
python examples/calibrate/auto_calibrate_example.py --port COM3 --device-type robot
```

Linux 示例：

```bash
python examples/calibrate/auto_calibrate_example.py --port /dev/ttyACM0 --device-type robot
```

## 图形界面

安装带 `tools` 可选依赖后，可直接运行：

```bash
lerobot-auto-calibrate-ui
```

也可以直接运行脚本：

```bash
python src/lerobot/scripts/auto_calibrate_ui.py
```

### 界面主要区域

- 设备类型选择：`tele` / `robot`
- 串口选择：自动检测本机可用串口
- 输出文件名：决定最终生成的 JSON 文件名
- 运行状态：显示当前阶段
- 日志输出：显示实时日志
- 输出路径：显示最终保存路径

### 按钮说明

- `开始标定`：启动自动标定
- `暂停标定`：暂停当前标定；暂停时当前动作会安全停下，舵机保持锁住，点击后可继续
- `重新标定`：在失败或完成后重新开始
- `检测机械臂`：驱动 `gripper` 或 6 号舵机执行双向边界检测
- `Linux 端口授权`：仅 Linux 显示，执行 `/dev/ttyACM*` 权限授权

## 机械臂检测

机械臂检测不会执行完整标定，它会：

- 连接当前选择的设备
- 找到 `gripper`，如果没有则尝试 6 号舵机
- 复用标定逻辑中的边界探索函数
- 朝一个方向运动直到碰到边界
- 停止后反向运动到另一侧边界

如果检测过程报错，界面会显示“检测出问题”，不会误报成标定失败。

## 暂停 / 继续

图形界面的暂停按钮支持暂停当前标定流程：

- 当前动作会先安全停下
- 舵机会保持锁住
- 点击“继续标定”后继续流程

暂停能力已经接入标定核心流程中的边界探索和等待环节。

## 输出路径

界面会显示本次标定结果的完整保存路径。

规则如下：

- 保存目录沿用设备默认 `calibration_fpath`
- 文件名由界面输入框决定
- 如果没有写 `.json`，程序会自动补上

## 默认参数

### tele

```python
try_torque = 400
max_torque = 500
torque_step = 50
explore_velocity = 600
wait_time_s = 0.5
velocity_threshold = 4
position_tolerance = 4000
```

### robot

```python
try_torque = 600
max_torque = 1000
torque_step = 50
explore_velocity = 800
wait_time_s = 0.5
velocity_threshold = 4
position_tolerance = 4000
```

## 安全建议

开始标定前建议确认：

- 机械臂周围没有障碍物
- 电源稳定
- 串口连接正常
- 知道如何快速断电
- 大负载关节先使用更保守参数

如果出现过流、撞边界过猛、卡死等现象，建议先降低：

- `try_torque`
- `max_torque`
- `explore_velocity`

## 常见问题

### Linux 下为什么看不到串口？

当前 UI 只显示 `/dev/ttyACM*`，请先确认设备枚举到该路径。

### Linux 下为什么打不开串口？

可以点击 `Linux 端口授权` 按钮，输入 sudo 密码后为 `/dev/ttyACM*` 添加读写权限。

### 标定结果保存在哪里？

界面右侧会直接显示本次输出路径；命令行模式下默认保存到设备自己的 calibration 路径。
