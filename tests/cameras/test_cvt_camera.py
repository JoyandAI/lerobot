#!/usr/bin/env python3
"""
Simple test script to verify CvtOpenCVCamera functionality.
"""

def test_config_creation():
    """Test CvtOpenCVCameraConfig creation."""
    try:
        from lerobot.cameras.opencv import CvtOpenCVCameraConfig
        
        # Test valid config
        config = CvtOpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=360,
            in_fps=30,
            in_width=1280,
            in_height=1024
        )
        print("✅ Config creation successful")
        print(f"   Input: {config.in_width}x{config.in_height}@{config.in_fps}fps")
        print(f"   Output: {config.width}x{config.height}@{config.fps}fps")
        return True
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

def test_camera_creation():
    """Test CvtOpenCVCamera creation (without connecting)."""
    try:
        from lerobot.cameras.opencv import CvtOpenCVCamera, CvtOpenCVCameraConfig
        
        config = CvtOpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=360,
            in_fps=30,
            in_width=1280,
            in_height=1024
        )
        
        camera = CvtOpenCVCamera(config)
        print("✅ Camera creation successful")
        print(f"   Camera: {camera}")
        print(f"   Width: {camera.width}, Height: {camera.height}")
        print(f"   Needs conversion: {camera.needs_conversion}")
        return True
        
    except Exception as e:
        print(f"❌ Camera creation failed: {e}")
        return False

def test_conversion_logic():
    """Test conversion logic with synthetic data."""
    try:
        from lerobot.cameras.opencv import CvtOpenCVCamera, CvtOpenCVCameraConfig
        import numpy as np
        
        config = CvtOpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=360,
            in_fps=30,
            in_width=1280,
            in_height=1024
        )
        
        camera = CvtOpenCVCamera(config)
        
        # Create synthetic frame
        synthetic_frame = np.random.randint(0, 255, (1024, 1280, 3), dtype=np.uint8)
        print(f"   Input frame shape: {synthetic_frame.shape}")
        
        # Test conversion
        converted_frame = camera._convert_frame(synthetic_frame)
        print(f"   Output frame shape: {converted_frame.shape}")
        
        expected_shape = (360, 640, 3)
        if converted_frame.shape == expected_shape:
            print("✅ Conversion logic successful")
            return True
        else:
            print(f"❌ Conversion failed: expected {expected_shape}, got {converted_frame.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Conversion test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing CvtOpenCVCamera...")
    print("=" * 50)
    
    tests = [
        ("Config Creation", test_config_creation),
        ("Camera Creation", test_camera_creation),
        ("Conversion Logic", test_conversion_logic),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed!")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CvtOpenCVCamera is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
