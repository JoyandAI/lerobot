# CvtOpenCVCamera 使用指南

## 概述

`CvtOpenCVCamera` 是一个通用的摄像头分辨率转换类，支持任意输入输出分辨率的转换。它继承自 `OpenCVCamera`，提供了智能的缩放和裁剪功能。

## 核心特性

### 🎯 智能缩放策略
- **缩放倍率选择**：自动选择更大的缩放倍率，即只裁剪，不填0
- **长宽比保持**：使用统一的缩放倍率保持长宽比一致

### 🔧 居中裁剪
- 当缩放后的尺寸与目标尺寸不完全匹配时，自动进行居中裁剪
- 确保输出尺寸精确匹配目标分辨率

### ⚡ 插值方法优化
- **降采样**：使用 `cv2.INTER_AREA`（更适合缩小）
- **上采样**：使用 `cv2.INTER_CUBIC`（更适合放大）

## 使用方法

### 基本配置

```python
from lerobot.cameras.opencv import CvtOpenCVCamera, CvtOpenCVCameraConfig
from lerobot.cameras.configuration_opencv import ColorMode

# 创建配置
config = CvtOpenCVCameraConfig(
    index_or_path=0,        # 摄像头索引或路径
    fps=30,                 # 输出帧率（必须与输入帧率相同）
    width=640,              # 输出宽度
    height=360,             # 输出高度
    in_fps=30,              # 输入帧率
    in_width=1280,          # 输入宽度
    in_height=1024,         # 输入高度
    color_mode=ColorMode.RGB
)

# 创建摄像头实例
camera = CvtOpenCVCamera(config)
camera.connect()

# 读取转换后的帧
frame = camera.read()  # 返回 (360, 640, 3) 的帧
```

### 转换示例

#### 示例1：1280x1024 → 640x360
```python
config = CvtOpenCVCameraConfig(
    index_or_path=0,
    fps=30, width=640, height=360
    in_fps=30, in_width=1280, in_height=1024,
)
```
- 缩放倍率：x=0.5, y=0.3516
- 选择倍率：0.5（更接近1.0）
- 过程：1280x1024 → 640x512 → 居中裁剪到 640x360

#### 示例2：640x480 → 1280x720
```python
config = CvtOpenCVCameraConfig(
    index_or_path=0,
    fps=30, width=1280, height=720
    in_fps=30, in_width=640, in_height=480,
)
```
- 缩放倍率：x=2.0, y=1.5
- 选择倍率：1.5（更接近1.0）
- 过程：640x480 → 960x720 → 居中裁剪到 1280x720

#### 示例3：1920x1080 → 640x360
```python
config = CvtOpenCVCameraConfig(
    index_or_path=0,
    fps=30, width=640, height=360
    in_fps=30, in_width=1920, in_height=1080,
)
```
- 缩放倍率：x=0.333, y=0.333
- 选择倍率：0.333（相同）
- 过程：1920x1080 → 640x360（直接缩放，无需裁剪）

## 配置参数

### CvtOpenCVCameraConfig 参数

| 参数 | 类型 | 描述 | 示例 |
|------|------|------|------|
| `index_or_path` | int/Path | 摄像头索引或设备路径 | `0`, `"/dev/video0"` |
| `fps` | int | 输出帧率（必须与in_fps相同） | `30` |
| `width` | int | 输出宽度 | `640` |
| `height` | int | 输出高度 | `360` |
| `in_fps` | int | 输入帧率 | `30` |
| `in_width` | int | 输入宽度 | `1280` |
| `in_height` | int | 输入高度 | `1024` |
| `color_mode` | ColorMode | 颜色模式 | `ColorMode.RGB` |
| `rotation` | Cv2Rotation | 图像旋转 | `Cv2Rotation.NO_ROTATION` |

## 转换算法详解

### 1. 缩放倍率计算
```python
scale_x = output_width / input_width
scale_y = output_height / input_height
```

### 2. 智能倍率选择
```python
if scale_x < 1.0 and scale_y < 1.0:
    # 都缩小：选择较大的倍率（更温和）
    scale_factor = max(scale_x, scale_y)
elif scale_x > 1.0 and scale_y > 1.0:
    # 都放大：选择较小的倍率（更温和）
    scale_factor = min(scale_x, scale_y)
else:
    # 混合：选择更接近1.0的倍率
    scale_factor = scale_x if abs(scale_x - 1.0) < abs(scale_y - 1.0) else scale_y
```

### 3. 转换过程
```python
# 步骤1：按选择的倍率缩放
scaled_width = int(input_width * scale_factor)
scaled_height = int(input_height * scale_factor)
scaled_frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation)

# 步骤2：居中裁剪到目标尺寸
if scaled_width != output_width or scaled_height != output_height:
    crop_x = (scaled_width - output_width) // 2
    crop_y = (scaled_height - output_height) // 2
    final_frame = scaled_frame[crop_y:crop_y+output_height, crop_x:crop_x+output_width]
```

## 性能考虑

### 内存使用
- 转换过程会创建中间帧，增加内存使用
- 建议在内存充足的环境中使用

### 计算开销
- 每次 `read()` 都会进行转换计算
- 对于实时应用，建议使用异步读取 `async_read()`

### 质量优化
- 降采样使用 `INTER_AREA` 插值，质量更好
- 上采样使用 `INTER_CUBIC` 插值，细节更丰富

## 错误处理

### 常见错误

1. **帧率不匹配**
   ```
   ValueError: Input fps (30) must match output fps (60)
   ```

2. **尺寸无效**
   ```
   ValueError: Input dimensions must be positive, got 0x0
   ```

3. **摄像头连接失败**
   ```
   DeviceNotConnectedError: Camera is not connected
   ```

## 最佳实践

1. **选择合适的输出分辨率**：避免过度缩放
2. **保持帧率一致**：目前只支持相同帧率
3. **测试不同配置**：根据具体摄像头特性调整参数
4. **监控性能**：在高帧率应用中注意CPU使用率

## 完整示例

```python
#!/usr/bin/env python3
import cv2
from lerobot.cameras.opencv import CvtOpenCVCamera, CvtOpenCVCameraConfig
from lerobot.cameras.configuration_opencv import ColorMode

def main():
    # 配置摄像头转换
    config = CvtOpenCVCameraConfig(
        index_or_path=0,
        in_fps=30, in_width=1280, in_height=1024,
        fps=30, width=640, height=360,
        color_mode=ColorMode.RGB
    )
    
    # 创建并连接摄像头
    camera = CvtOpenCVCamera(config)
    camera.connect()
    
    print(f"Camera: {camera}")
    print(f"Input: {config.in_width}x{config.in_height}@{config.in_fps}fps")
    print(f"Output: {config.width}x{config.height}@{config.fps}fps")
    
    try:
        while True:
            # 读取转换后的帧
            frame = camera.read()
            
            # 显示帧信息
            print(f"\rFrame shape: {frame.shape}", end="", flush=True)
            
            # 显示图像
            cv2.imshow('Converted Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
