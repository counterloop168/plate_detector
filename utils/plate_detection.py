"""
License Plate Detection Module
Handles YOLO-based plate detection and OCR for Vietnamese license plates
Designed specifically for Camera 1 (IN) and Camera 2 (OUT)
"""

import os
import sys
import cv2
import torch
import numpy as np
import json
import logging
from threading import Lock
from datetime import datetime
from collections import defaultdict, deque
from fast_alpr import ALPR
from fast_alpr.base import BaseDetector, DetectionResult
from ultralytics import YOLO

# Import functions from lp_detection_video.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from lp_detection_video import (
        validate_vietnamese_plate_format,
        format_plate,
        extract_crop,
        run_detection_on_crops,
        draw_detection_box,
        draw_crop_windows
    )
    LP_VIDEO_FUNCTIONS_AVAILABLE = True
except ImportError:
    LP_VIDEO_FUNCTIONS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)  # Changed from DEBUG to INFO
logger = logging.getLogger(__name__)

if not LP_VIDEO_FUNCTIONS_AVAILABLE:
    logger.warning("Could not import functions from lp_detection_video.py - using fallback implementations")

# Configuration
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'plate_detection_config.json')
config = {}

def load_config():
    """Load plate detection configuration"""
    global config
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded plate detection config: {config}")
        return True
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Default config
        config = {
            "enabled_cameras": [1, 2],
            "detection_fps": 1,
            "confidence_threshold": 0.6,
            "deduplication": {
                "time_window_seconds": 5,
                "hash_threshold": 0.95,
                "min_votes_required": 3
            },
            "models": {
                "detector_path": "utils/models/best.pt",
                "ocr_model": "cct-s-v1-global-model"
            },
            "crop_config": {
                "num_crops": 2,
                "crop_size": 640,
                "x_position": 600,
                "y_position": 0
            }
        }
        return False

# Load configuration on module import
load_config()

# Global model instances
plate_detector = None
alpr = None
model_lock = Lock()

# Tracking state
tracked_plates = {}
plate_id_counter = 0
plate_state_lock = Lock()

# Last detection time per camera (for FPS throttling)
last_detection_time = {}
detection_time_lock = Lock()


class CustomYOLODetector(BaseDetector):
    """Custom YOLO detector for license plates"""
    
    def __init__(self, model_path: str):
        try:
            self.model = YOLO(model_path)
            
            # Check GPU availability
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.model.to('cuda')
                logger.info(f"Custom YOLO detector loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                logger.info("Custom YOLO detector loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load custom YOLO model: {e}")
            raise

    def predict(self, frame: np.ndarray) -> list:
        """Run detection on frame"""
        try:
            results = self.model(
                frame, 
                conf=0.25, 
                iou=0.4, 
                verbose=False,
                device=self.device,
                half=True if self.device == 'cuda' else False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        detection = DetectionResult(
                            bounding_box=type('BoundingBox', (), {
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                            })(),
                            confidence=confidence,
                            label="license_plate"
                        )
                        detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []


def init_plate_detection_models():
    """Initialize plate detection and OCR models"""
    global plate_detector, alpr
    
    with model_lock:
        if plate_detector is not None and alpr is not None:
            return True
            
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                config['models']['detector_path']
            )
            
            if not os.path.exists(model_path):
                logger.error(f"Plate detection model not found: {model_path}")
                return False
            
            logger.info(f"Loading plate detection model from {model_path}")
            plate_detector = CustomYOLODetector(model_path)
            
            # Initialize ALPR with custom detector
            ocr_model = config['models']['ocr_model']
            alpr = ALPR(
                detector=plate_detector,
                ocr_model=ocr_model
            )
            
            logger.info("Plate detection models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize plate detection models: {e}")
            plate_detector = None
            alpr = None
            return False


# Fallback implementations if lp_detection_video.py functions are not available
if not LP_VIDEO_FUNCTIONS_AVAILABLE:
    def validate_vietnamese_plate_format(text):
        """Validate Vietnamese license plate format (fallback)"""
        if not text:
            return False
        
        clean_text = text.replace(' ', '').replace('-', '').replace('.', '').upper()
        
        if len(clean_text) < 8 or len(clean_text) > 10:
            return False
        
        # Position 1-2: digits, Position 3: letter, Position 4: digit/letter, Position 5+: digits
        if not (clean_text[0].isdigit() and clean_text[1].isdigit()):
            return False
        if not clean_text[2].isalpha():
            return False
        if not (clean_text[3].isdigit() or clean_text[3].isalpha()):
            return False
        
        for i in range(4, len(clean_text)):
            if not clean_text[i].isdigit():
                return False
        
        return True

    def format_plate(text):
        """Format plate text with standard Vietnamese format (fallback)"""
        if not text:
            return ""
        
        clean_text = text.replace(' ', '').replace('-', '').replace('.', '').upper()
        
        if len(clean_text) == 8 or len(clean_text) == 9:
            first_three = clean_text[:3]
            last = clean_text[3:]
            return f"{first_three}-{last}"
        return clean_text


def get_detection_region(frame, camera_id):
    """Get region of interest for plate detection based on camera ID"""
    h, w = frame.shape[:2]
    
    # Camera 1 (IN): Detection region matching red box position (center-right area)
    if camera_id == 1:
        x_start = 300  # Red box position from screenshot
        y_start = 140  # Red box position from screenshot
        
        # Ensure we get exactly 640x640 by adjusting if needed
        if y_start + 640 > h:
            y_start = max(0, h - 640)  # Adjust to fit within frame
        if x_start + 640 > w:
            x_start = max(0, w - 640)  # Adjust to fit within frame
            
        x_end = x_start + 640
        y_end = y_start + 640
        
        crop = frame[y_start:y_end, x_start:x_end]
        return crop, (x_start, y_start)
    
    # Camera 2 (OUT): Detection region matching red box position (center-left area)
    elif camera_id == 2:
        x_start = 200  # Red box position from screenshot
        y_start = 140  # Red box position from screenshot
        
        # Ensure we get exactly 640x640 by adjusting if needed
        if y_start + 640 > h:
            y_start = max(0, h - 640)  # Adjust to fit within frame
        if x_start + 640 > w:
            x_start = max(0, w - 640)  # Adjust to fit within frame
            
        x_end = x_start + 640
        y_end = y_start + 640
        
        crop = frame[y_start:y_end, x_start:x_end]
        return crop, (x_start, y_start)
    
    # For other cameras, return full frame
    return frame, (0, 0)


def distance(box1, box2):
    """Calculate distance between two bounding box centers"""
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def update_text_history(plate_info, new_text):
    """Update text history and accept any text immediately (no validation, no voting system)"""
    if 'text_history' not in plate_info:
        plate_info['text_history'] = []
    if 'best_text' not in plate_info:
        plate_info['best_text'] = ''
    if 'best_confidence' not in plate_info:
        plate_info['best_confidence'] = 0
    
    # Accept ANY text from OCR - no format validation
    # Critical for poor quality cameras and non-standard plates
    if new_text and len(new_text.strip()) > 0:
        # Clean up the text
        clean_text = new_text.strip().upper()
        
        plate_info['text_history'].append(clean_text)
        
        # Keep only last 10 detections for reference
        if len(plate_info['text_history']) > 10:
            plate_info['text_history'] = plate_info['text_history'][-10:]
        
        # REMOVED VOTING SYSTEM: Accept immediately with 100% confidence
        # REMOVED FORMAT VALIDATION: Accept any text
        plate_info['best_text'] = clean_text
        plate_info['best_confidence'] = 100  # Mark as confident
    
    return plate_info.get('best_text', '')


def calculate_text_confidence(plate_info, text):
    """Calculate confidence based on frequency in text history"""
    text_history = plate_info.get('text_history', [])
    if not text_history or not text:
        return 0
    
    text_count = text_history.count(text)
    total_count = len(text_history)
    
    confidence = (text_count / total_count) * 100
    return confidence


def update_tracking(camera_id, detections):
    """Update tracking using distance-based matching"""
    global plate_id_counter, tracked_plates
    
    tracking_threshold = 50  # Distance threshold for tracking
    
    with plate_state_lock:
        # Get camera-specific tracked plates
        camera_key = str(camera_id)
        if camera_key not in tracked_plates:
            tracked_plates[camera_key] = {}
        
        camera_plates = tracked_plates[camera_key]
        current_plates = {}
        
        for detection in detections:
            x1, y1, x2, y2 = detection
            best_match_id = None
            min_distance = float('inf')
            
            # Find best match with existing tracked plates
            for plate_id, plate_info in camera_plates.items():
                dist = distance(detection, plate_info['bbox'])
                if dist < tracking_threshold and dist < min_distance:
                    min_distance = dist
                    best_match_id = plate_id
            
            if best_match_id is not None:
                # Update existing plate
                camera_plates[best_match_id]['bbox'] = detection
                camera_plates[best_match_id]['frames_since_seen'] = 0
                camera_plates[best_match_id]['frame_count'] += 1
                current_plates[best_match_id] = camera_plates[best_match_id]
            else:
                # New plate detected
                plate_id_counter += 1
                new_plate = {
                    'id': plate_id_counter,
                    'bbox': detection,
                    'frames_since_seen': 0,
                    'saved': False,
                    'text': '',
                    'confidence': 0.0,
                    'text_history': [],
                    'frame_count': 1,
                    'best_text': '',
                    'best_confidence': 0,
                    'camera_id': camera_id,
                    'first_seen': datetime.now()
                }
                camera_plates[plate_id_counter] = new_plate
                current_plates[plate_id_counter] = new_plate
        
        # Update frames_since_seen for plates not detected in this frame
        plates_to_remove = []
        for plate_id, plate_info in camera_plates.items():
            if plate_id not in current_plates:
                camera_plates[plate_id]['frames_since_seen'] += 1
                # Remove plates not seen for more than 60 frames
                if camera_plates[plate_id]['frames_since_seen'] > 60:
                    plates_to_remove.append(plate_id)
        
        # Remove old plates
        for plate_id in plates_to_remove:
            del camera_plates[plate_id]
        
        return current_plates


def process_frame_full(frame, camera_id):
    """
    Process frame with region-based detection for better plate detection
    Returns: (processed_frame, detected_plates_info)
    """
    if alpr is None:
        logger.warning(f"[Cam {camera_id}] ALPR not initialized")
        return frame, []
    
    try:
        result_frame = frame.copy()
        detections = []
        alpr_data = {}
        
        # Get region of interest for detection
        detection_region, offset = get_detection_region(frame, camera_id)
        x_offset, y_offset = offset
        
        # Run ALPR detection on the cropped region
        alpr_results = alpr.predict(detection_region)
        
        if alpr_results:
            for result in (alpr_results if isinstance(alpr_results, list) else [alpr_results]):
                if hasattr(result, 'detection') and hasattr(result.detection, 'bounding_box'):
                    bbox = result.detection.bounding_box
                    # Adjust coordinates back to full frame
                    x1, y1, x2, y2 = int(bbox.x1) + x_offset, int(bbox.y1) + y_offset, int(bbox.x2) + x_offset, int(bbox.y2) + y_offset
                    
                    # Validate bounding box is within frame
                    if (x1 < frame.shape[1] and y1 < frame.shape[0] and 
                        x2 < frame.shape[1] and y2 < frame.shape[0] and
                        x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1):
                        
                        text = result.ocr.text if hasattr(result, 'ocr') and result.ocr.text else ""
                        confidence = result.ocr.confidence if hasattr(result, 'ocr') and result.ocr.confidence else 0.0
                        
                        detection = [x1, y1, x2, y2]
                        detections.append(detection)
                        alpr_data[str(detection)] = {'text': text, 'confidence': confidence}
        
        # Update tracking
        current_plates = update_tracking(camera_id, detections)
        
        # Draw detection region rectangle
        if x_offset > 0 or y_offset > 0:
            region_h, region_w = detection_region.shape[:2]
            cv2.rectangle(result_frame, (x_offset, y_offset), 
                         (x_offset + region_w, y_offset + region_h), (255, 0, 255), 3)
            cv2.putText(result_frame, "Detection Region", 
                       (x_offset + 10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # Draw detection status
        status_y = 30
        region_name = "Bottom-Right" if camera_id == 1 else "Bottom-Left" if camera_id == 2 else "Full Frame"
        cv2.putText(result_frame, f"Cam {camera_id} | Region Detection ({region_name})", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Frame: {frame.shape[1]}x{frame.shape[0]} | Region: {detection_region.shape[1]}x{detection_region.shape[0]} | Detections: {len(detections)}",
                   (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Process and draw results
        detected_plates_info = []
        
        for plate_id, plate_info in current_plates.items():
            x1, y1, x2, y2 = plate_info['bbox']
            
            # Get ALPR text for this detection
            detection_key = str([x1, y1, x2, y2])
            if detection_key in alpr_data:
                recognized_text = alpr_data[detection_key]['text']
                if recognized_text:
                    with plate_state_lock:
                        camera_key = str(camera_id)
                        display_text = update_text_history(tracked_plates[camera_key][plate_id], recognized_text)
                        tracked_plates[camera_key][plate_id]['text'] = display_text
                        plate_info['text'] = display_text
            
            # Get text confidence
            display_text = plate_info.get('best_text', '')
            confidence_percentage = int(plate_info.get('best_confidence', 0))
            
            # Choose color - all plates with text are green (no validation, no voting system)
            if display_text:
                color = (0, 255, 0)  # Green for any detected plate
                
                # Add to detected plates info
                detected_plates_info.append({
                    'plate_number': display_text,
                    'confidence': confidence_percentage,
                    'bbox': [x1, y1, x2, y2],
                    'plate_id': plate_info['id'],
                    'frame_count': plate_info.get('frame_count', 0),
                    'camera_id': camera_id
                })
            else:
                color = (0, 255, 255)  # Yellow for reading
                display_text = "Reading..."
            
            # Draw bounding box with thicker lines
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 4)
            
            # Draw label with background for better visibility
            label = f"ID:{plate_info['id']} {display_text}"
            if confidence_percentage > 0:
                label += f" ({confidence_percentage}%)"
            
            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result_frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            cv2.putText(result_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Only log when plates are actually detected
        if detected_plates_info:
            logger.info(f"[Cam {camera_id}] Detected {len(detected_plates_info)} plates: {[p['plate_number'] for p in detected_plates_info]}")
        
        return result_frame, detected_plates_info
        
    except Exception as e:
        logger.error(f"[Cam {camera_id}] Error processing frame: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return frame with error message
        error_frame = frame.copy()
        cv2.putText(error_frame, f"Detection Error: {str(e)[:50]}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_frame, []


def detect_plates(frame, camera_id):
    """
    Main entry point for plate detection
    Throttles detection to configured FPS
    Returns: (processed_frame, detected_plates_info)
    """
    global last_detection_time
    
    # Check if camera is enabled for plate detection
    if camera_id not in config['enabled_cameras']:
        logger.warning(f"[Cam {camera_id}] Not enabled for plate detection")
        disabled_frame = frame.copy()
        cv2.putText(disabled_frame, f"Cam {camera_id}: Plate detection disabled", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return disabled_frame, []
    
    # Check FPS throttling
    current_time = datetime.now().timestamp()
    skip_detection = False
    with detection_time_lock:
        last_time = last_detection_time.get(camera_id, 0)
        min_interval = 1.0 / config['detection_fps']
        
        if current_time - last_time < min_interval:
            # Skip detection, but still show visualization
            skip_detection = True
        else:
            last_detection_time[camera_id] = current_time
    
    if skip_detection:
        # Return frame with status overlay
        throttled_frame = frame.copy()
        cv2.putText(throttled_frame, f"Cam {camera_id}: Waiting for next detection cycle", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return throttled_frame, []
    
    # Initialize models if needed
    if alpr is None:
        logger.info(f"[Cam {camera_id}] Initializing ALPR models...")
        if not init_plate_detection_models():
            error_frame = frame.copy()
            cv2.putText(error_frame, f"Cam {camera_id}: Model initialization failed", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return error_frame, []
        logger.info(f"[Cam {camera_id}] Models initialized successfully")
    
    # Process frame
    return process_frame_full(frame, camera_id)


def get_tracked_plates(camera_id=None):
    """Get tracked plates for a specific camera or all cameras"""
    with plate_state_lock:
        if camera_id is not None:
            camera_key = str(camera_id)
            return tracked_plates.get(camera_key, {})
        return tracked_plates


def reset_tracking(camera_id=None):
    """Reset tracking state for a camera or all cameras"""
    with plate_state_lock:
        if camera_id is not None:
            camera_key = str(camera_id)
            if camera_key in tracked_plates:
                tracked_plates[camera_key] = {}
        else:
            tracked_plates.clear()


# Initialize on module import
logger.info("Plate detection module loaded")
