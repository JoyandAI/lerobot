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

"""
Provides the CvtOpenCVCamera class for capturing frames from cameras using OpenCV with resolution conversion.
"""

import logging
import time
from typing import Any

import cv2
import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .camera_opencv import OpenCVCamera
from .configuration_opencv import CvtOpenCVCameraConfig

logger = logging.getLogger(__name__)


class CvtOpenCVCamera(OpenCVCamera):
    """
    OpenCV camera with resolution and frame rate conversion capabilities.
    
    This class extends OpenCVCamera to provide resolution and frame rate conversation.

    Conversion process:
    1. Capture at in_width * in_height @ in_fps
    2. Center crop to width*height (centering)
    3. Downsample to width*height
    4. Return width*height @ fps frames
    
    Example:
        ```python
        from lerobot.cameras.opencv import CvtOpenCVCamera
        from lerobot.cameras.configuration_opencv import OpenCVCameraConfig, ColorMode
        
        # The camera will be detected and converted automatically
        cvt_config = CvtOpenCVCameraConfig(
            in_fps=30,
            in_width=1280,
            in_height=1024,
            index_or_path=0,
            fps=30,
            width=640,
            height=360,
        )
        camera = CvtOpenCVCamera(cvt_config)
        camera.connect()
        
        # Read converted frame (width*height)
        frame = camera.read()
        print(frame.shape)  # (height, width, 3)
        ```
    """

    def __init__(self, config: CvtOpenCVCameraConfig):
        """
        Initialize the CvtOpenCVCamera instance.
        
        Args:
            config: The configuration settings for the camera.
        """
        # Store input and output dimensions BEFORE any modifications
        self.input_width = config.in_width
        self.input_height = config.in_height
        self.input_fps = config.in_fps

        # Store output dimensions in private variables to prevent modification
        self._output_width = config.width
        self._output_height = config.height
        
        # Flag to prevent setter from modifying _output_width/_output_height during parent init
        self._initializing = True

        # Temporarily modify config for parent initialization
        # Parent OpenCVCamera will use in_width and in_height for actual capture
        original_width = config.width
        original_height = config.height
        config.width = config.in_width
        config.height = config.in_height
        
        # Call parent initialization with modified config
        # pdb.set_trace()
        super().__init__(config)
        
        # Restore original config values
        config.width = original_width
        config.height = original_height
        
        # Initialization complete, enable setters
        self._initializing = False
        
        # Check if this camera needs conversion
        self.needs_conversion = self._should_convert()
        
        if self.needs_conversion:
            logger.info(f"Camera {self.index_or_path} will do conversion: {self.input_width}x{self.input_height}@{self.input_fps} -> {self._output_width}x{self._output_height}@{self.fps}")
            # Calculate scaling factors
            self.scale_x = self._output_width * 1.0 / self.input_width
            self.scale_y = self._output_height * 1.0 / self.input_height
            self.scale_factor = self._calculate_scale_factor()
            logger.info(f"Scale factors: x={self.scale_x:.3f}, y={self.scale_y:.3f}, using={self.scale_factor:.3f}")
        else:
            logger.info(f"Camera {self.index_or_path} will not do conversion")

    def _should_convert(self) -> bool:
        """
        Check if this camera should do resolution and frame rate conversion.
        
        Returns:
            True if camera needs conversion (in_width*in_height@in_fps != width*height@fps)
        """
        return (self.input_width != self._output_width or
                self.input_height != self._output_height or
                self.input_fps != self.fps)
    
    def _calculate_scale_factor(self) -> float:
        """
        Calculate the optimal scale factor for conversion.
        
        Choose the larger scale factor (just crop, no padding):
        
        Returns:
            The scale factor
        """
        return max(self.scale_x, self.scale_y)

    def _convert_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame from in_width*in_height@in_fps to width*height@fps.
        
        Args:
            frame: Input frame of shape (in_height, in_width, 3)
            
        Returns:
            Converted frame of shape (height, width, 3)
        """
        if not self.needs_conversion:
            return frame
        
        # Step 1: Scale using the calculated scale factor
        scaled_width = int(self.input_width * self.scale_factor)
        scaled_height = int(self.input_height * self.scale_factor)
        
        # Choose interpolation method based on scaling direction
        if self.scale_factor < 1.0:
            interpolation = cv2.INTER_AREA  # Better for downscaling
        else:
            interpolation = cv2.INTER_CUBIC  # Better for upscaling
        
        scaled_frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=interpolation)
        
        # Step 2: Center crop to exact output dimensions
        if scaled_width != self._output_width or scaled_height != self._output_height:
            # Calculate crop start positions for centering
            crop_start_x = (scaled_width - self._output_width) // 2
            crop_start_y = (scaled_height - self._output_height) // 2
            crop_end_x = crop_start_x + self._output_width
            crop_end_y = crop_start_y + self._output_height
            
            # Ensure crop coordinates are within bounds
            crop_start_x = max(0, crop_start_x)
            crop_start_y = max(0, crop_start_y)
            crop_end_x = min(scaled_width, crop_end_x)
            crop_end_y = min(scaled_height, crop_end_y)
            
            converted_frame = scaled_frame[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        else:
            converted_frame = scaled_frame
        
        return converted_frame

    def read(self) -> np.ndarray:
        """
        Read a single frame from the camera with resolution conversion if needed.
        
        Returns:
            Frame as numpy array. Shape will be (360, 640, 3) if conversion is enabled,
            otherwise original resolution.
            
        Raises:
            DeviceNotConnectedError: If camera is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Read frame at original resolution
        frame = super().read()
        
        # Apply conversion if needed
        if self.needs_conversion:
            frame = self._convert_frame(frame)
            
        return frame

    def async_read(self) -> np.ndarray:
        """
        Read a single frame asynchronously with resolution conversion if needed.
        
        Returns:
            Frame as numpy array. Shape will be (360, 640, 3) if conversion is enabled,
            otherwise original resolution.
            
        Raises:
            DeviceNotConnectedError: If camera is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Read frame at original resolution
        frame = super().async_read()
        
        # Apply conversion if needed
        if self.needs_conversion:
            frame = self._convert_frame(frame)
            
        return frame

    @property   
    def width(self) -> int:
        if hasattr(self, '_initializing') and self._initializing:
            return self.input_width
        return self._output_width

    @width.setter
    def width(self, value: int):
        if not hasattr(self, '_initializing') or not self._initializing:
            self._output_width = value

    @property
    def height(self) -> int:
        if hasattr(self, '_initializing') and self._initializing:
            return self.input_height
        return self._output_height

    @height.setter
    def height(self, value: int):
        if not hasattr(self, '_initializing') or not self._initializing:
            self._output_height = value

    def configure(self):
        """Configure camera to use input dimensions for capture."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Cannot configure settings for {self} as it is not connected.")

        # Get actual camera dimensions
        default_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Use input dimensions for capture
        self.capture_width = self.input_width
        self.capture_height = self.input_height
        
        # Apply rotation if needed
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.input_height, self.input_width

        # Validate fourcc
        self._validate_fourcc()
        
        # Set fps if not already set
        if self.fps is None:
            self.fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        else:
            self.videocapture.set(cv2.CAP_PROP_FPS, self.fps)

        # Set resolution to input dimensions
        if (self.capture_width != default_width or 
            self.capture_height != default_height):
            self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

    def __str__(self) -> str:
        """String representation of the camera."""
        conversion_info = " (converted)" if self.needs_conversion else ""
        return f"{self.__class__.__name__}({self.index_or_path}){conversion_info}"
