# Auto Calibrate 使用说明

这个文档说明如何使用 [auto_calibrate.py](/d:/workspace/lerobot/tools/auto_calibrate.py) 和 [auto_calibrate_example.py](/d:/workspace/lerobot/tools/auto_calibrate_example.py) 对 SO101 设备执行自动标定。

当前示例脚本已经改成命令行参数模式，支持两类设备：

- `tele`: SO101 leader / teleoperator
- `robot`: SO101 follower / robot

## 功能概览

自动标定脚本会按当前实现流程完成以下工作：

- 连接设备
- 逐个关节探索机械边界
- 计算 `homing_offset`、`range_min`、`range_max`
- 生成 `MotorCalibration`
- 保存为 calibration JSON 文件

## 相关文件

- [auto_calibrate.py](/d:/workspace/lerobot/tools/auto_calibrate.py): 自动标定核心逻辑
- [auto_calibrate_example.py](/d:/workspace/lerobot/tools/auto_calibrate_example.py): 命令行入口示例

## 命令行参数

必选参数：

- `--port`: 串口号，例如 `COM9`
- `--device-type`: 设备类型，只能是 `tele` 或 `robot`

可选参数：

- `--id`: 设备 ID，默认是 `my_so101`
- `--calibration-dir`: 指定标定文件输出目录
- `--try-torque`: 覆盖默认 `try_torque`
- `--max-torque`: 覆盖默认 `max_torque`
- `--torque-step`: 覆盖默认 `torque_step`
- `--explore-velocity`: 覆盖默认 `explore_velocity`
- `--wait-time-s`: 覆盖默认 `wait_time_s`
- `--velocity-threshold`: 覆盖默认 `velocity_threshold`
- `--position-tolerance`: 覆盖默认 `position_tolerance`
- `--log`: 是否在终端输出 logger，默认 `true`，例如 `--log=false`
- `--yes`: 跳过开始前的回车确认

## 默认参数

### `tele` 默认参数

```python
try_torque = 400
max_torque = 500
torque_step = 50
explore_velocity = 600
wait_time_s = 0.5
velocity_threshold = 4
position_tolerance = 4000
```

### `robot` 默认参数

```python
try_torque = 600
max_torque = 1000
torque_step = 50
explore_velocity = 600
wait_time_s = 0.5
velocity_threshold = 4
position_tolerance = 4000
```

## 终端输入示例

### 1. 标定 tele

```bash
python tools/auto_calibrate_example.py --port COM9 --device-type tele
```

### 2. 标定 robot

```bash
python tools/auto_calibrate_example.py --port COM3 --device-type robot
```

### 3. 指定设备 ID

```bash
python tools/auto_calibrate_example.py --port COM9 --device-type tele --id my_so101
```

### 4. 覆盖默认扭矩参数

```bash
python tools/auto_calibrate_example.py --port COM9 --device-type tele --try-torque 300 --max-torque 450
```

### 5. 覆盖多个参数

```bash
python tools/auto_calibrate_example.py --port COM3 --device-type robot --try-torque 500 --max-torque 800 --explore-velocity 500 --wait-time-s 0.7
```

### 6. 跳过确认提示

```bash
python tools/auto_calibrate_example.py --port COM9 --device-type tele --yes
```

### 7. 关闭终端 logger 输出

```bash
python tools/auto_calibrate_example.py --port COM9 --device-type tele --log=false
```

## 执行流程

运行脚本后，大致流程如下：

1. 根据 `--device-type` 创建对应配置对象
2. `tele` 使用 `SO101LeaderConfig`
3. `robot` 使用 `SO101FollowerConfig`
4. 根据设备类型加载对应默认标定参数
5. 如果命令行给了可选参数，则覆盖默认值
6. 默认会在终端输出 logger，传 `--log=false` 可关闭
7. 连接设备并执行自动标定
8. 保存标定结果到设备对应的 calibration 文件路径

## 输出结果

标定成功后，终端会输出每个电机的结果，例如：

```text
电机: shoulder_pan
  ID: 1
  Homing Offset: -1396
  Range Min: 620
  Range Max: 3440
```

保存后的 JSON 结构类似：

```json
{
  "shoulder_pan": {
    "id": 1,
    "drive_mode": 0,
    "homing_offset": -1396,
    "range_min": 620,
    "range_max": 3440
  }
}
```

## 常见说明

### `MotorCalibration` 是什么

标定函数最终会返回 `MotorCalibration` 对象，内部字段包括：

- `id`
- `drive_mode`
- `homing_offset`
- `range_min`
- `range_max`

访问方式例如：

```python
print(calibration.id)
print(calibration.homing_offset)
print(calibration.range_min)
print(calibration.range_max)
```

### `tele` 和 `robot` 的区别

- `tele` 默认扭矩更保守
- `robot` 默认扭矩更大，适合 follower 负载场景

### `start_torque`

当前版本的自动标定逻辑里，主要实际使用的是：

- `try_torque`
- `max_torque`
- `torque_step`

如果你在调参数，优先关注这几个。

## 安全建议

开始标定前建议确认：

- 机械臂周围没有障碍物
- 电源稳定
- 串口连接正常
- 知道如何快速断电
- 肩部和肘部等大负载关节先用更保守参数测试

如果出现类似 `OverEle error`、过流、卡死、持续顶住边界等现象，先降低：

- `try_torque`
- `max_torque`
- `explore_velocity`

## 推荐调参起点

如果设备容易过流或撞边界太猛，可以先尝试：

### 对 `tele`

```bash
python tools/auto_calibrate_example.py --port COM9 --device-type tele --try-torque 300 --max-torque 400 --explore-velocity 400
```

### 对 `robot`

```bash
python tools/auto_calibrate_example.py --port COM3 --device-type robot --try-torque 400 --max-torque 600 --explore-velocity 400
```

## 备注

这个 README 以当前仓库中的 [auto_calibrate_example.py](/d:/workspace/lerobot/tools/auto_calibrate_example.py) 为准。

如果你后续又修改了脚本参数，README 也需要同步更新。
