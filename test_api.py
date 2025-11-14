"""
Test script for Plate Detection Web Server API
Tests all endpoints to ensure they're working correctly
"""

import requests
import json
import os
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5000"

def print_result(test_name, success, message=""):
    """Print test result"""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"      {message}")
    print()

def test_health_check():
    """Test health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        data = response.json()
        success = response.status_code == 200 and data.get('status') == 'ok'
        message = f"Status: {data.get('status')}, Models: {data.get('models_loaded')}"
        print_result("Health Check", success, message)
        return success
    except Exception as e:
        print_result("Health Check", False, str(e))
        return False

def test_get_config():
    """Test get configuration endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/config", timeout=5)
        data = response.json()
        success = response.status_code == 200 and data.get('success')
        message = f"Enabled cameras: {data.get('config', {}).get('enabled_cameras', [])}"
        print_result("Get Configuration", success, message)
        return success
    except Exception as e:
        print_result("Get Configuration", False, str(e))
        return False

def test_init_models():
    """Test model initialization endpoint"""
    try:
        response = requests.post(f"{BASE_URL}/api/init-models", timeout=30)
        data = response.json()
        success = data.get('success', False)
        message = data.get('message', data.get('error', ''))
        print_result("Initialize Models", success, message)
        return success
    except Exception as e:
        print_result("Initialize Models", False, str(e))
        return False

def test_recent_detections():
    """Test recent detections endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/recent-detections?minutes=5", timeout=5)
        data = response.json()
        success = response.status_code == 200 and data.get('success')
        message = f"Found {data.get('count', 0)} recent detections"
        print_result("Recent Detections", success, message)
        return success
    except Exception as e:
        print_result("Recent Detections", False, str(e))
        return False

def test_tracking_stats():
    """Test tracking statistics endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/tracking-stats", timeout=5)
        data = response.json()
        success = response.status_code == 200 and data.get('success')
        stats = data.get('stats', {})
        message = f"Total cached: {stats.get('total_cached_plates', 0)}"
        print_result("Tracking Stats", success, message)
        return success
    except Exception as e:
        print_result("Tracking Stats", False, str(e))
        return False

def test_clear_cache():
    """Test clear cache endpoint"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/clear-cache",
            json={},
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        data = response.json()
        success = response.status_code == 200 and data.get('success')
        message = data.get('message', '')
        print_result("Clear Cache", success, message)
        return success
    except Exception as e:
        print_result("Clear Cache", False, str(e))
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Plate Detection Web Server API Tests")
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
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Get Configuration", test_get_config),
        ("Initialize Models", test_init_models),
        ("Recent Detections", test_recent_detections),
        ("Tracking Statistics", test_tracking_stats),
        ("Clear Cache", test_clear_cache),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
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
    else:
        print(f"⚠️ {total - passed} test(s) failed")
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
