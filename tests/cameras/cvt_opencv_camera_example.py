#!/usr/bin/env python3
"""
Example script demonstrating the CvtOpenCVCamera class usage.

This script shows how to use the CvtOpenCVCamera class to automatically
convert camera resolution from 1280x1024 to 640x360 for model inference.
"""

import cv2
import numpy as np
from lerobot.cameras.opencv import CvtOpenCVCamera, CvtOpenCVCameraConfig
from lerobot.cameras.opencv.configuration_opencv import ColorMode

import matplotlib.pyplot as plt

def display_frame(frame):
    """使用 matplotlib 显示帧，替代 cv2.imshow"""
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Camera Feed')
    plt.pause(0.001)  # 短暂暂停以更新显示
    plt.clf()  # 清除当前图形

def main():
    """Main function demonstrating CvtOpenCVCamera usage."""
    
    # Create configuration for camera with resolution conversion
    config = CvtOpenCVCameraConfig(
        index_or_path=0,  # Camera index
        fps=30,
        width=800,
        height=450,
        in_fps=30,
        in_width=3840,
        in_height=2160, 
    )
    
    # Create camera instance
    camera = CvtOpenCVCamera(config)
    
    try:
        # Connect to camera
        print("Connecting to camera...")
        camera.connect()
        print(f"Camera connected: {camera}")
        print(f"Output resolution: {camera.width}x{camera.height}")
        
        # Read and display frames
        print("Reading frames (press 'q' to quit)...")
        frame_count = 0
        
        while True:
            # Read frame (will be automatically converted to 640x360)
            frame = camera.read()
            frame_count += 1
            
            # Display frame info
            print(f"\rFrame {frame_count}: {frame.shape} - Press 'q' to quit", end="", flush=True)
            
            # Display frame using OpenCV
            display_frame(frame)
            
            # Check for quit key
            if plt.waitforbuttonpress(0.001):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Clean up
        camera.disconnect()
        # cv2.destroyAllWindows()
        print("\nCamera disconnected and cleaned up")


def test_conversion():
    """Test the conversion functionality with synthetic frames."""
    
    print("Testing conversion functionality...")
    
    # Test case 1: 1280x1024 -> 640x360 (downscaling)
    print("\nTest 1: 1280x1024 -> 640x360")
    synthetic_frame_1 = np.random.randint(0, 255, (1024, 1280, 3), dtype=np.uint8)
    
    config_1 = CvtOpenCVCameraConfig(
        index_or_path=0,
        fps=30,
        width=640,
        height=360,
        in_fps=30,
        in_width=1280,
        in_height=1024,
    )
    
    camera_1 = CvtOpenCVCamera(config_1)
    converted_frame_1 = camera_1._convert_frame(synthetic_frame_1)
    
    print(f"Original: {synthetic_frame_1.shape}")
    print(f"Converted: {converted_frame_1.shape}")
    print(f"Expected: (360, 640, 3)")
    assert converted_frame_1.shape == (360, 640, 3), f"Expected (360, 640, 3), got {converted_frame_1.shape}"
    print("✅ Test 1 passed!")

    # Test case 2: 1280x720 -> 640x360 (downscaling)
    print("\nTest 2: 1280x720 -> 640x360")
    synthetic_frame_2 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    config_2 = CvtOpenCVCameraConfig(
        index_or_path=0,
        fps=30,
        width=640,
        height=360,
        in_fps=30,
        in_width=1280,
        in_height=720,
    )
    
    camera_2 = CvtOpenCVCamera(config_2)
    converted_frame_2 = camera_2._convert_frame(synthetic_frame_2)
    
    print(f"Original: {synthetic_frame_2.shape}")
    print(f"Converted: {converted_frame_2.shape}")
    print(f"Expected: (360, 640, 3)")
    assert converted_frame_2.shape == (360, 640, 3), f"Expected (360, 640, 3), got {converted_frame_2.shape}"
    print("✅ Test 2 passed!")
    
    # Test case 3: 640x480 -> 1280x720 (upscaling and different aspect ratio)
    print("\nTest 3: 640x480 -> 1280x720 (upscaling and different aspect ratio)")
    synthetic_frame_3 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    config_3 = CvtOpenCVCameraConfig(
        index_or_path=0,
        fps=30,
        width=1280,
        height=720,
        in_fps=30,
        in_width=640,
        in_height=480,
    )
    
    camera_3 = CvtOpenCVCamera(config_3)
    converted_frame_3 = camera_3._convert_frame(synthetic_frame_3)
    
    print(f"Original: {synthetic_frame_3.shape}")
    print(f"Converted: {converted_frame_3.shape}")
    print(f"Expected: (720, 1280, 3)")
    assert converted_frame_3.shape == (720, 1280, 3), f"Expected (720, 1280, 3), got {converted_frame_3.shape}"
    print("✅ Test 3 passed!")
    
    print("\n🎉 All conversion tests passed!")


if __name__ == "__main__":
    print("CvtOpenCVCamera Example")
    print("=" * 50)
    
    # Test conversion functionality first
    # test_conversion()
    print()
    
    # Run main example
    main()
