"""
Test script for Real-Time Camera Detection
Tests camera endpoints and functionality
"""

import requests
import time
import sys

BASE_URL = "http://localhost:5000"

def print_result(test_name, success, message=""):
    """Print test result"""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"      {message}")
    print()

def test_camera_status():
    """Test camera status endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/camera/status", timeout=5)
        data = response.json()
        success = response.status_code == 200 and data.get('success')
        message = f"Cameras: {data.get('cameras', {})}"
        print_result("Camera Status Check", success, message)
        return success
    except Exception as e:
        print_result("Camera Status Check", False, str(e))
        return False

def test_start_camera(camera_id=1, camera_source=0):
    """Test starting camera"""
    try:
        print(f"Starting Camera {camera_id} with source {camera_source}...")
        response = requests.post(
            f"{BASE_URL}/api/camera/start",
            json={'camera_id': camera_id, 'camera_source': camera_source},
            timeout=30
        )
        data = response.json()
        success = data.get('success', False)
        message = data.get('message', data.get('error', ''))
        if success:
            message += f" | Stream URL: {data.get('stream_url')}"
        print_result(f"Start Camera {camera_id}", success, message)
        return success
    except Exception as e:
        print_result(f"Start Camera {camera_id}", False, str(e))
        return False

def test_camera_stream(camera_id=1, duration=5):
    """Test camera stream (check if accessible)"""
    try:
        print(f"Testing Camera {camera_id} stream for {duration} seconds...")
        stream_url = f"{BASE_URL}/api/camera/stream/{camera_id}?source=0"
        
        # Try to access stream
        response = requests.get(stream_url, stream=True, timeout=10)
        
        if response.status_code == 200:
            # Read a few frames
            bytes_received = 0
            start_time = time.time()
            
            for chunk in response.iter_content(chunk_size=1024):
                bytes_received += len(chunk)
                if time.time() - start_time > duration:
                    break
            
            success = bytes_received > 0
            message = f"Received {bytes_received:,} bytes in {duration} seconds"
            print_result(f"Camera {camera_id} Stream", success, message)
            return success
        else:
            print_result(f"Camera {camera_id} Stream", False, f"HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_result(f"Camera {camera_id} Stream", False, str(e))
        return False

def test_stop_camera(camera_id=1):
    """Test stopping camera"""
    try:
        print(f"Stopping Camera {camera_id}...")
        response = requests.post(
            f"{BASE_URL}/api/camera/stop",
            json={'camera_id': camera_id},
            timeout=10
        )
        data = response.json()
        success = data.get('success', False)
        message = data.get('message', data.get('error', ''))
        print_result(f"Stop Camera {camera_id}", success, message)
        return success
    except Exception as e:
        print_result(f"Stop Camera {camera_id}", False, str(e))
        return False

def test_camera_page():
    """Test camera page loads"""
    try:
        response = requests.get(f"{BASE_URL}/camera", timeout=5)
        success = response.status_code == 200 and 'Live Camera Feed' in response.text
        message = f"HTTP {response.status_code}"
        print_result("Camera Page Load", success, message)
        return success
    except Exception as e:
        print_result("Camera Page Load", False, str(e))
        return False

def main():
    """Run all camera tests"""
    print("=" * 60)
    print("Real-Time Camera Detection Tests")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print()
    
    # Check if server is running
    try:
        requests.get(BASE_URL, timeout=2)
    except requests.exceptions.RequestException:
        print("✗ Server is not running!")
        print(f"Please start the server first: python app.py")
        print()
        return
    
    print("✓ Server is running")
    print()
    
    # Get camera source from user
    print("Camera Source Options:")
    print("  0 - Default camera (webcam)")
    print("  1 - Second camera")
    print("  2 - Third camera")
    print("  Or enter RTSP URL (e.g., rtsp://192.168.1.100:554/stream)")
    print()
    
    camera_source = input("Enter camera source (default: 0): ").strip()
    if not camera_source:
        camera_source = 0
    else:
        try:
            camera_source = int(camera_source)
        except ValueError:
            pass  # Keep as string for RTSP URL
    
    print()
    print("Starting tests...")
    print()
    
    # Run tests
    results = []
    
    # Test 1: Camera page loads
    results.append(("Camera Page", test_camera_page()))
    
    # Test 2: Camera status check
    results.append(("Camera Status", test_camera_status()))
    
    # Test 3: Start camera
    camera_started = test_start_camera(1, camera_source)
    results.append(("Start Camera", camera_started))
    
    if camera_started:
        # Wait a moment for camera to initialize
        print("Waiting for camera to initialize...")
        time.sleep(3)
        print()
        
        # Test 4: Stream test
        results.append(("Camera Stream", test_camera_stream(1, duration=5)))
        
        # Test 5: Stop camera
        results.append(("Stop Camera", test_stop_camera(1)))
    else:
        print("⚠️ Camera failed to start. Skipping stream tests.")
        print("   This might be normal if:")
        print("   - No camera is connected")
        print("   - Camera is in use by another app")
        print("   - Wrong camera source specified")
        print()
        results.append(("Camera Stream", False))
        results.append(("Stop Camera", False))
    
    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print()
    
    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")
    
    print()
    
    if passed == total:
        print("🎉 All tests passed!")
        print()
        print("Next steps:")
        print("1. Open http://localhost:5000/camera in your browser")
        print("2. Click 'Start Camera' to begin live detection")
        print("3. Watch for plate detections in real-time")
    elif camera_started:
        print("⚠️ Some tests failed, but camera is working!")
        print("You can still use the camera at: http://localhost:5000/camera")
    else:
        print("⚠️ Camera tests failed.")
        print()
        print("Troubleshooting:")
        print("1. Check if camera is connected")
        print("2. Try different camera source (0, 1, 2)")
        print("3. Close other apps using the camera")
        print("4. Check camera permissions")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
