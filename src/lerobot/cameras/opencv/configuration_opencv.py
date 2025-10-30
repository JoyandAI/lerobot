# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """Configuration class for OpenCV-based camera devices or video files.

    This class provides configuration options for cameras accessed through OpenCV,
    supporting both physical camera devices and video files. It includes settings
    for resolution, frame rate, color mode, and image rotation.

    Example configurations:
    ```python
    # Basic configurations
    OpenCVCameraConfig(0, 30, 1280, 720)   # 1280x720 @ 30FPS
    OpenCVCameraConfig(/dev/video4, 60, 640, 480)   # 640x480 @ 60FPS

    # Advanced configurations
    OpenCVCameraConfig(128422271347, 30, 640, 480, rotation=Cv2Rotation.ROTATE_90)     # With 90° rotation
    ```

    Attributes:
        index_or_path: Either an integer representing the camera device index,
                      or a Path object pointing to a video file.
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)

    Note:
        - Only 3-channel color output (RGB/BGR) is currently supported.
    """

    index_or_path: int | Path
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self):
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )


@CameraConfig.register_subclass("cvt_opencv")
@dataclass
class CvtOpenCVCameraConfig(OpenCVCameraConfig):
    """Configuration class for OpenCV camera with resolution and frame rate conversion.
    
    This extends OpenCVCameraConfig to add flexible resolution and frame rate conversion.
    The camera will capture at in_width*in_height@in_fps and convert to width*height@fps.
    
    Attributes:
        in_fps: Input frame rate (must match fps for now)
        in_width: Input frame width
        in_height: Input frame height
        index_or_path: Either an integer representing the camera device index,
                      or a Path object pointing to a video file.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        fps: Output frame rate (must match in_fps for now)
        width: Output frame width
        height: Output frame height
        
    Example:
        ```python
        # Camera with resolution conversion: 1280x1024@30fps -> 640x360@30fps
        config = CvtOpenCVCameraConfig(
            in_fps=30,
            in_width=1280,
            in_height=1024,
            index_or_path=0,
            fps=30,
            width=640,
            height=360,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
            warmup_s=1,
        )
        ```
    """

    in_fps: int | None = None
    in_width: int | None = None
    in_height: int | None = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()
        
        # Check if input parameters are provided
        if self.in_fps is None or self.in_width is None or self.in_height is None:
            raise ValueError(
                "Input parameters (in_fps, in_width, in_height) are required for CvtOpenCVCameraConfig"
            )
        
        # Validate frame rate consistency
        if self.in_fps != self.fps:
            raise ValueError(
                f"Input fps ({self.in_fps}) must match output fps ({self.fps}). "
                "Frame rate conversion is not yet supported."
            )
        
        # Validate input dimensions
        if self.in_width <= 0 or self.in_height <= 0:
            raise ValueError(f"Input dimensions must be positive, got {self.in_width}x{self.in_height}")
        
        # Validate output dimensions
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Output dimensions must be positive, got {self.width}x{self.height}")