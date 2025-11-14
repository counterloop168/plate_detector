#!/usr/bin/env python3
"""
Plate Detection Visualization Test Script
This script helps visualize and debug the plate detection crop regions
"""

import cv2
import json
import numpy as np
from utils.plate_detection import detect_plates, init_plate_detection_models

def load_config():
    """Load plate detection configuration"""
    with open('utils/plate_detection_config.json', 'r') as f:
        return json.load(f)

def draw_crop_regions(frame, config, camera_id):
    """Draw crop regions on frame for visualization"""
    camera_key = str(camera_id)
    if 'camera_specific_crops' in config and camera_key in config['camera_specific_crops']:
        crop_config = config['camera_specific_crops'][camera_key]
    else:
        crop_config = config['crop_config']
    
    x = crop_config['x_position']
    y = crop_config['y_position']
    crop_size = crop_config['crop_size']
    num_crops = crop_config['num_crops']
    
    result = frame.copy()
    
    for i in range(num_crops):
        crop_x = x
        crop_y = y + crop_size * i
        
        color = [(255, 0, 255), (0, 255, 255), (255, 255, 0)][i % 3]
        
        # Draw crop rectangle
        cv2.rectangle(result, 
                     (crop_y, crop_x), 
                     (min(crop_y + crop_size, frame.shape[1]), min(crop_x + crop_size, frame.shape[0])),
                     color, 3)
        cv2.putText(result, f"Crop {i+1}", (crop_y + 10, crop_x + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Add frame info
    cv2.putText(result, f"Frame: {frame.shape[1]}x{frame.shape[0]}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f"Camera {camera_id}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result

def test_static_image(image_path, camera_id):
    """Test detection on a static image"""
    print(f"\n=== Testing on static image: {image_path} ===")
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not load image from {image_path}")
        return
    
    print(f"Image loaded: {frame.shape}")
    
    # Initialize models
    print("Initializing plate detection models...")
    if not init_plate_detection_models():
        print("ERROR: Failed to initialize models")
        return
    print("Models initialized successfully")
    
    # Load config
    config = load_config()
    
    # Draw crop regions
    crop_viz = draw_crop_regions(frame, config, camera_id)
    cv2.imwrite(f"/tmp/crop_regions_cam{camera_id}.jpg", crop_viz)
    print(f"Saved crop visualization to /tmp/crop_regions_cam{camera_id}.jpg")
    
    # Run detection
    print("Running detection...")
    detected_frame, plates_info = detect_plates(frame, camera_id)
    
    print(f"\nDetection Results:")
    print(f"  Total plates detected: {len(plates_info)}")
    for plate in plates_info:
        print(f"    - Plate: {plate['plate_number']}, Confidence: {plate['confidence']}%, BBox: {plate['bbox']}")
    
    # Save result
    cv2.imwrite(f"/tmp/detection_result_cam{camera_id}.jpg", detected_frame)
    print(f"Saved detection result to /tmp/detection_result_cam{camera_id}.jpg")
    
    return detected_frame, plates_info

def adjust_crop_positions_interactive(camera_id, rtsp_url=None):
    """
    Interactive tool to adjust crop positions
    Use keyboard to move crop regions:
    - Arrow keys: move crop region
    - +/-: change crop size
    - 's': save configuration
    - 'q': quit
    """
    print(f"\n=== Interactive Crop Adjustment for Camera {camera_id} ===")
    print("Controls:")
    print("  Arrow keys: Move crop region")
    print("  +/-: Change crop size")
    print("  's': Save configuration")
    print("  'q': Quit")
    
    if rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
    else:
        # Use default camera
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    config = load_config()
    camera_key = str(camera_id)
    
    # Get or create camera-specific config
    if 'camera_specific_crops' not in config:
        config['camera_specific_crops'] = {}
    
    if camera_key not in config['camera_specific_crops']:
        config['camera_specific_crops'][camera_key] = config['crop_config'].copy()
    
    crop_config = config['camera_specific_crops'][camera_key]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            break
        
        # Draw crop regions
        viz_frame = draw_crop_regions(frame, config, camera_id)
        
        # Show current config
        cv2.putText(viz_frame, f"x={crop_config['x_position']}, y={crop_config['y_position']}, size={crop_config['crop_size']}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f'Camera {camera_id} - Crop Adjustment', viz_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save configuration
            with open('utils/plate_detection_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            print(f"\nConfiguration saved!")
            print(f"  x_position: {crop_config['x_position']}")
            print(f"  y_position: {crop_config['y_position']}")
            print(f"  crop_size: {crop_config['crop_size']}")
        elif key == 82:  # Up arrow
            crop_config['x_position'] = max(0, crop_config['x_position'] - 10)
        elif key == 84:  # Down arrow
            crop_config['x_position'] = min(frame.shape[0] - crop_config['crop_size'], crop_config['x_position'] + 10)
        elif key == 81:  # Left arrow
            crop_config['y_position'] = max(0, crop_config['y_position'] - 10)
        elif key == 83:  # Right arrow
            crop_config['y_position'] = min(frame.shape[1] - crop_config['crop_size'], crop_config['y_position'] + 10)
        elif key == ord('+') or key == ord('='):
            crop_config['crop_size'] = min(1920, crop_config['crop_size'] + 20)
        elif key == ord('-') or key == ord('_'):
            crop_config['crop_size'] = max(320, crop_config['crop_size'] - 20)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Plate Detection Visualization Test Tool")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python test_plate_visualization.py <camera_id> [image_path]")
        print("\nExamples:")
        print("  python test_plate_visualization.py 1 test_image.jpg")
        print("  python test_plate_visualization.py 2")
        sys.exit(1)
    
    camera_id = int(sys.argv[1])
    
    if len(sys.argv) >= 3:
        # Test on static image
        image_path = sys.argv[2]
        test_static_image(image_path, camera_id)
    else:
        # Interactive adjustment
        print("\nNo image provided, using live camera feed...")
        adjust_crop_positions_interactive(camera_id)
