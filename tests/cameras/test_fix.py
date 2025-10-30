#!/usr/bin/env python3
"""
Test the fix for CvtOpenCVCamera.
"""

def test_camera_creation():
    try:
        print("Testing CvtOpenCVCamera creation...")
        
        from lerobot.cameras.opencv import CvtOpenCVCamera, CvtOpenCVCameraConfig
        
        # Create config
        config = CvtOpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=360,
            in_fps=30,
            in_width=1280,
            in_height=1024
        )
        print("✅ Config created")
        
        # Create camera
        camera = CvtOpenCVCamera(config)
        print("✅ Camera created")
        print(f"   Width: {camera.width}")
        print(f"   Height: {camera.height}")
        print(f"   Needs conversion: {camera.needs_conversion}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_camera_creation()
    if success:
        print("\n🎉 Test passed!")
    else:
        print("\n❌ Test failed!")
