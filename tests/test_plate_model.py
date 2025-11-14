import sys
from fast_alpr import ALPR
from fast_alpr.base import BaseDetector, DetectionResult
import cv2
import os
import torch
import ultralytics
import numpy as np
from collections import defaultdict
import hashlib
from datetime import datetime

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ Ultralytics not available")
    sys.exit(1)

# Custom detector class that extends BaseDetector
class CustomYOLODetector(BaseDetector):
    def __init__(self, model_path: str):
        try:
            self.model = YOLO(model_path)
            
            # Check GPU availability and set device
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.model.to('cuda')
                print(f"✅ Custom YOLO detector loaded from: {model_path}")
                print(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                self.device = 'cpu'
                print(f"✅ Custom YOLO detector loaded from: {model_path}")
                print("⚠️  GPU not available, running on CPU")
                
        except Exception as e:
            print(f"❌ Failed to load custom YOLO model: {e}")
            raise

    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        # Run inference on GPU with optimized settings
        results = self.model(
            frame, 
            conf=0.5, 
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
                    # Get bounding box coordinates (ensure CPU conversion)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Create DetectionResult with required label parameter
                    detection = DetectionResult(
                        bounding_box=type('BoundingBox', (), {
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                        })(),
                        confidence=confidence,
                        label="license_plate"  
                    )
                    detections.append(detection)
        
        return detections

try:
    print("🔄 Attempting to load custom detector model...")
    custom_detector = CustomYOLODetector(r"C:\Users\Admin\Desktop\License_plate_regconition\plate_11m\runs\detect\train\weights\best.pt")
    
    alpr = ALPR(
        detector=custom_detector, 
        ocr_model="cct-s-v1-global-model", 
    )
    print("✅ Custom detector model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load custom detector model: {e}")
    print("🚪 Exiting program...")
    sys.exit(1)

# Global tracking variables (using video_fast_alpr_2.py logic)
tracked_plates = {}
plate_id_counter = 0
saved_plates = set()
tracking_threshold = 50  # Distance threshold for tracking

def validate_vietnamese_plate_format(text):
    """Validate Vietnamese license plate format"""
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
    """Format plate text with standard Vietnamese format"""
    if not text:
        return ""
    
    clean_text = text.replace(' ', '').replace('-', '').replace('.', '').upper()
    
    if len(clean_text) == 8 or len(clean_text) == 9:
        first_three = clean_text[:3]  
        last = clean_text[3:]     
        return f"{first_three}-{last}"
    return clean_text

def distance(box1, box2):
    """Calculate distance between two bounding box centers"""
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def update_text_history(plate_info, new_text):
    """Update text history and determine best text based on frequency and confidence"""
    if 'text_history' not in plate_info:
        plate_info['text_history'] = []
    if 'best_text' not in plate_info:
        plate_info['best_text'] = ''
    if 'best_confidence' not in plate_info:
        plate_info['best_confidence'] = 0
    
    if new_text and validate_vietnamese_plate_format(new_text):
        formatted_text = format_plate(new_text)
        clean_length = len(formatted_text.replace(' ', '').replace('-', '').replace('.', ''))
        if clean_length >= 8 and clean_length <= 10:
            plate_info['text_history'].append(formatted_text)
            
            # Keep only last 30 detections for optimal performance and accuracy
            if len(plate_info['text_history']) > 30:
                plate_info['text_history'] = plate_info['text_history'][-30:]
            
            current_confidence = calculate_text_confidence(plate_info, formatted_text)
            if current_confidence > plate_info['best_confidence']:
                plate_info['best_text'] = formatted_text
                plate_info['best_confidence'] = current_confidence
    
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

def get_text_confidence_info(plate_info):
    """Get best text and confidence information"""
    best_text = plate_info.get('best_text', '')
    best_confidence = plate_info.get('best_confidence', 0)
    text_history = plate_info.get('text_history', [])
    
    if not best_text:
        return '', 0, len(text_history)
    
    frequency = text_history.count(best_text) if text_history else 0
    total_detections = len(text_history)
    
    return best_text, frequency, total_detections

def update_tracking(detections):
    """Update tracking using distance-based matching (from video_fast_alpr_2.py)"""
    global plate_id_counter, tracked_plates
    
    current_plates = {}
    
    for detection in detections:
        x1, y1, x2, y2 = detection
        best_match_id = None
        min_distance = float('inf')
        
        # Find best match with existing tracked plates
        for plate_id, plate_info in tracked_plates.items():
            dist = distance(detection, plate_info['bbox'])
            if dist < tracking_threshold and dist < min_distance:
                min_distance = dist
                best_match_id = plate_id
        
        if best_match_id is not None:
            # Update existing plate
            tracked_plates[best_match_id]['bbox'] = detection
            tracked_plates[best_match_id]['frames_since_seen'] = 0
            tracked_plates[best_match_id]['frame_count'] += 1
            current_plates[best_match_id] = tracked_plates[best_match_id]
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
                'best_confidence': 0
            }
            tracked_plates[plate_id_counter] = new_plate
            current_plates[plate_id_counter] = new_plate
    
    # Update frames_since_seen for plates not detected in this frame
    plates_to_remove = []
    for plate_id, plate_info in tracked_plates.items():
        if plate_id not in current_plates:
            tracked_plates[plate_id]['frames_since_seen'] += 1
            # Remove plates not seen for more than 60 frames
            if tracked_plates[plate_id]['frames_since_seen'] > 60:
                plates_to_remove.append(plate_id)
    
    # Remove old plates
    for plate_id in plates_to_remove:
        del tracked_plates[plate_id]
    
    return current_plates

def draw_crop_windows(frame, x, y, times, crop_size, show_detection=True):
    """Draw crop windows on frame for visualization"""
    overlay = frame.copy()
    
    for i in range(times):
        # Calculate crop coordinates
        crop_x = x
        crop_y = y + crop_size * i
        
        # Ensure crop doesn't exceed frame boundaries for visualization
        display_width = min(crop_size, frame.shape[1] - crop_y)
        display_height = min(crop_size, frame.shape[0] - crop_x)
        
        if display_width <= 0 or display_height <= 0:
            continue
        
        # Choose different colors for each crop window
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        color = colors[i % len(colors)]
        
        # Draw main crop rectangle (only visible part)
        cv2.rectangle(overlay, (crop_y, crop_x), (crop_y + display_width, crop_x + display_height), color, 3)
        
        # Draw corner markers for better visibility
        corner_size = 25
        corner_thickness = 5
        
        # Top-left corner
        cv2.line(overlay, (crop_y, crop_x), (crop_y + min(corner_size, display_width), crop_x), color, corner_thickness)
        cv2.line(overlay, (crop_y, crop_x), (crop_y, crop_x + min(corner_size, display_height)), color, corner_thickness)
        
        # Top-right corner
        cv2.line(overlay, (crop_y + display_width, crop_x), (crop_y + display_width - min(corner_size, display_width), crop_x), color, corner_thickness)
        cv2.line(overlay, (crop_y + display_width, crop_x), (crop_y + display_width, crop_x + min(corner_size, display_height)), color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(overlay, (crop_y, crop_x + display_height), (crop_y + min(corner_size, display_width), crop_x + display_height), color, corner_thickness)
        cv2.line(overlay, (crop_y, crop_x + display_height), (crop_y, crop_x + display_height - min(corner_size, display_height)), color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(overlay, (crop_y + display_width, crop_x + display_height), (crop_y + display_width - min(corner_size, display_width), crop_x + display_height), color, corner_thickness)
        cv2.line(overlay, (crop_y + display_width, crop_x + display_height), (crop_y + display_width, crop_x + display_height - min(corner_size, display_height)), color, corner_thickness)
        
        # Add crop label with background
        label = f"Crop {i+1}"
        if i == 1:  # Second crop with padding
            label += " (+Padding)"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label at top of crop window
        label_x = crop_y + 10
        label_y = crop_x - 10 if crop_x > 50 else crop_x + display_height + text_height + 10
        
        # Draw label background
        # cv2.rectangle(overlay, 
        #              (label_x, label_y - text_height - 10), 
        #              (label_x + text_width + 20, label_y + 10), 
        #              color, -1)
        
        # # Draw label text
        # cv2.putText(overlay, label, (label_x + 10, label_y), 
        #            font, font_scale, (255, 255, 255), thickness)
        
        # Add coordinates info
        # coord_info = f"({crop_y},{crop_x})"
        # coord_font_scale = 0.8
        # cv2.putText(overlay, coord_info, (label_x + 10, label_y + 25), 
        #            cv2.FONT_HERSHEY_SIMPLEX, coord_font_scale, (255, 255, 255), 2)
        
        # # Add size info
        # size_info = f"{display_width}x{display_height}"
        # if i == 1 and display_width < crop_size:
        #     size_info += f" (padded to {crop_size}x{crop_size})"
        # cv2.putText(overlay, size_info, (label_x + 10, label_y + 45), 
        #            cv2.FONT_HERSHEY_SIMPLEX, coord_font_scale, (255, 255, 255), 2)
    
    # Blend overlay with original frame
    alpha = 0.7
    result = cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0)
    
    return result

def process_frame_with_crops(frame, x, y, times, crop_size, show_crop_windows=True):
    """Process frame with crop detection, tracking, and draw results"""
    result_frame = frame.copy()
    
    # First, draw crop windows if requested
    if show_crop_windows:
        result_frame = draw_crop_windows(result_frame, x, y, times, crop_size)
    
    # Collect all detections from crops
    detections = []
    alpr_data = {}
    
    for i in range(times):
        # Calculate crop coordinates
        crop_x = x
        crop_y = y + crop_size * i
        
        # Extract crop for detection
        if i == 0:
            # First crop - normal extraction
            if crop_y + crop_size <= frame.shape[1] and crop_x + crop_size <= frame.shape[0]:
                crop_frame = frame[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size].copy()
            else:
                continue  # Skip if crop exceeds boundaries
        else:
            # Second crop - with black padding if needed
            # Calculate available area
            available_width = frame.shape[1] - crop_y
            available_height = frame.shape[0] - crop_x
            
            if available_width <= 0 or available_height <= 0:
                continue  # Skip if completely outside frame
            
            # Create 640x640 crop with black padding
            crop_frame = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            
            # Extract what we can from the original frame
            actual_width = min(available_width, crop_size)
            actual_height = min(available_height, crop_size)
            
            if actual_width > 0 and actual_height > 0:
                actual_crop = frame[crop_x:crop_x + actual_height, crop_y:crop_y + actual_width]
                # Place in top-left of the 640x640 crop
                crop_frame[:actual_height, :actual_width] = actual_crop
                
                print(f"📐 Crop {i+1} padding: {actual_width}x{actual_height} → 640x640 (added {640-actual_width}x{640-actual_height} black)")
        
        # Run ALPR detection on the crop
        alpr_results = alpr.predict(crop_frame)
        
        if alpr_results:
            # Process each detection and prepare for tracking
            for j, result in enumerate(alpr_results if isinstance(alpr_results, list) else [alpr_results]):
                if hasattr(result, 'detection') and hasattr(result.detection, 'bounding_box'):
                    bbox = result.detection.bounding_box
                    # Coordinates relative to crop
                    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                    
                    # Convert to original frame coordinates
                    orig_x1 = crop_y + x1
                    orig_y1 = crop_x + y1
                    orig_x2 = crop_y + x2
                    orig_y2 = crop_x + y2
                    
                    # Only include detection if it's within original frame boundaries
                    if (orig_x1 < frame.shape[1] and orig_y1 < frame.shape[0] and 
                        orig_x2 < frame.shape[1] and orig_y2 < frame.shape[0] and
                        orig_x1 >= 0 and orig_y1 >= 0):
                        
                        # Get OCR text and confidence
                        text = result.ocr.text if hasattr(result, 'ocr') and result.ocr.text else ""
                        confidence = result.ocr.confidence if hasattr(result, 'ocr') and result.ocr.confidence else 0.0
                        
                        detection = [orig_x1, orig_y1, orig_x2, orig_y2]
                        detections.append(detection)
                        
                        # Store OCR results
                        alpr_data[str(detection)] = {'text': text, 'confidence': confidence}
    
    # Update tracking with detections
    current_plates = update_tracking(detections)
    
    # Process each tracked plate and draw results
    for plate_id, plate_info in current_plates.items():
        x1, y1, x2, y2 = plate_info['bbox']
        
        # Get ALPR text for this detection
        detection_key = str([x1, y1, x2, y2])
        if detection_key in alpr_data:
            recognized_text = alpr_data[detection_key]['text']
            if recognized_text:
                display_text = update_text_history(tracked_plates[plate_id], recognized_text)
                tracked_plates[plate_id]['text'] = display_text
                plate_info['text'] = display_text
        
        # Get text confidence information
        display_text, frequency, total_detections = get_text_confidence_info(tracked_plates[plate_id])
        
        # Draw bounding box and text
        if display_text and validate_vietnamese_plate_format(display_text):
            confidence_percentage = int(tracked_plates[plate_id].get('best_confidence', 0))
            
            # Choose color based on confidence
            if confidence_percentage > 70:
                color = (0, 255, 0)  # Green for high confidence
                status = "VALID"
            elif confidence_percentage > 40:
                color = (0, 255, 255)  # Yellow for medium confidence
                status = "VALID (Med)"
            else:
                color = (255, 255, 0)  # Cyan for low confidence
                status = "VALID (Low)"
        else:
            # Yellow for plates being read/invalid
            color = (0, 255, 255)
            status = "READING"
            display_text = "Reading..."
            confidence_percentage = 0
        
        # Draw bounding box
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 4)
        
        # Draw corner markers for better visibility
        corner_size = 20
        corner_thickness = 4
        # Top-left
        cv2.line(result_frame, (x1, y1), (x1 + corner_size, y1), color, corner_thickness)
        cv2.line(result_frame, (x1, y1), (x1, y1 + corner_size), color, corner_thickness)
        # Top-right
        cv2.line(result_frame, (x2, y1), (x2 - corner_size, y1), color, corner_thickness)
        cv2.line(result_frame, (x2, y1), (x2, y1 + corner_size), color, corner_thickness)
        # Bottom-left
        cv2.line(result_frame, (x1, y2), (x1 + corner_size, y2), color, corner_thickness)
        cv2.line(result_frame, (x1, y2), (x1, y2 - corner_size), color, corner_thickness)
        # Bottom-right
        cv2.line(result_frame, (x2, y2), (x2 - corner_size, y2), color, corner_thickness)
        cv2.line(result_frame, (x2, y2), (x2, y2 - corner_size), color, corner_thickness)
        
        # Draw tracking ID and text
        label = f"ID:{plate_info['id']} {display_text}"
        confidence_label = f"Conf: {confidence_percentage}% ({frequency}/{total_detections})"
        frame_label = f"Frames: {plate_info['frame_count']}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text sizes
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (conf_width, conf_height), _ = cv2.getTextSize(confidence_label, font, font_scale * 0.8, thickness)
        (frame_width, frame_height), _ = cv2.getTextSize(frame_label, font, font_scale * 0.6, thickness)
        
        # Position text above bounding box if there's space
        if y1 > text_height + conf_height + frame_height + 50:
            text_x = x1
            text_y = y1 - conf_height - frame_height - 30
            conf_y = y1 - frame_height - 15
            frame_y = y1 - 5
        else:
            text_x = x1
            text_y = y2 + text_height + 20
            conf_y = y2 + text_height + conf_height + 30
            frame_y = y2 + text_height + conf_height + frame_height + 40
        
        # Draw text backgrounds
        # cv2.rectangle(result_frame, 
        #              (text_x, text_y - text_height - 10), 
        #              (text_x + text_width + 20, text_y + 10), 
        #              (0, 0, 0), -1)
        
        # cv2.rectangle(result_frame, 
        #              (text_x, conf_y - conf_height - 10), 
        #              (text_x + conf_width + 20, conf_y + 10), 
        #              (0, 0, 0), -1)
        
        # cv2.rectangle(result_frame, 
        #              (text_x, frame_y - frame_height - 10), 
        #              (text_x + frame_width + 20, frame_y + 10), 
        #              (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(result_frame, label, (text_x + 10, text_y), 
                   font, font_scale, color, thickness)
        cv2.putText(result_frame, confidence_label, (text_x + 10, conf_y), 
                   font, font_scale * 0.8, color, thickness)
        # cv2.putText(result_frame, frame_label, (text_x + 10, frame_y), 
        #            font, font_scale * 0.6, color, thickness)
    
    # Calculate statistics
    active_count = len(current_plates)
    total_tracked = len(tracked_plates)
    confirmed_count = len([p for p in tracked_plates.values() if p.get('best_text') and validate_vietnamese_plate_format(p.get('best_text'))])
    
    # Add summary overlay with tracking info
    summary = f"Active: {active_count} | Total: {total_tracked} | Confirmed: {confirmed_count}"
    position_info = f"Position: x={x}, y={y}, size={crop_size}x{crop_size}"
    tracking_info = f"Tracking threshold: {tracking_threshold}px"
    
    font_scale_summary = 1.0
    
    (summary_width, summary_height), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, font_scale_summary, 3)
    (pos_width, pos_height), _ = cv2.getTextSize(position_info, cv2.FONT_HERSHEY_SIMPLEX, font_scale_summary * 0.8, 2)
    (track_width, track_height), _ = cv2.getTextSize(tracking_info, cv2.FONT_HERSHEY_SIMPLEX, font_scale_summary * 0.6, 2)
    
    # Position summary at top-left
    max_width = max(summary_width, pos_width, track_width)
    # cv2.rectangle(result_frame, (20, 20), (30 + max_width, 110), (0, 0, 0), -1)
    # cv2.putText(result_frame, summary, (25, 45), 
    #            cv2.FONT_HERSHEY_SIMPLEX, font_scale_summary, (0, 255, 255), 3)
    # cv2.putText(result_frame, position_info, (25, 70), 
    #            cv2.FONT_HERSHEY_SIMPLEX, font_scale_summary * 0.8, (255, 255, 255), 2)
    # cv2.putText(result_frame, tracking_info, (25, 90), 
    #            cv2.FONT_HERSHEY_SIMPLEX, font_scale_summary * 0.6, (128, 128, 128), 2)
    
    return result_frame, active_count, confirmed_count

def main():
    """Main function to process video with crop detection, tracking, and visualization"""
    # Input and output video paths
    input_video = r"C:\Users\Admin\Desktop\License_plate_regconition\video_demo\input\demo_2.mp4"
    output_dir = r"C:\Users\Admin\Desktop\License_plate_regconition\video_demo\output\custom_detector_video"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Setup output video writer - KEEP ORIGINAL DIMENSIONS
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    output_video = os.path.join(output_dir, f"{base_name}_two_crops_with_padding.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))  # Original dimensions
    
    if not out.isOpened():
        print(f"❌ Error: Could not create output video writer")
        cap.release()
        return
    
    print(f"🚀 Starting video processing with two crop windows...")
    print(f"📁 Output will be saved to: {output_video}")
    
    # Crop parameters
    x = 600  # row position
    y = 0    # column position
    times = 2  # number of crop windows
    crop_size = 640  # size of each crop window
    
    print(f"📐 Crop parameters:")
    print(f"   🎯 Position: x={x} (row), y={y} (column)")
    print(f"   📊 Windows: {times}")
    print(f"   📏 Size: {crop_size}x{crop_size}")
    print(f"   🟢 Crop 1: (0-640, 600-1240) - normal extraction")
    print(f"   🔴 Crop 2: (640-1280, 600-1240) - with black padding for missing 200px")
    print(f"🎯 Tracking parameters:")
    print(f"   📊 Tracking threshold: {tracking_threshold}px")
    print(f"   📊 Max disappeared frames: 60")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame with crop detection, tracking, and visualization
            processed_frame, active_objects, confirmed_plates = process_frame_with_crops(
                frame, x, y, times, crop_size, show_crop_windows=True
            )
            
            # Write processed frame to output video (ORIGINAL DIMENSIONS)
            out.write(processed_frame)
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                total_tracked = len(tracked_plates)
                print(f"🔄 Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) - "
                      f"Active: {active_objects}, Total: {total_tracked}, Confirmed: {confirmed_plates}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
    finally:
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Get final statistics
    total_tracked_final = len(tracked_plates)
    confirmed_final = len([p for p in tracked_plates.values() if p.get('best_text') and validate_vietnamese_plate_format(p.get('best_text'))])
    
    # Print final summary
    print(f"\n📊 VIDEO PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"🎬 Input video: {os.path.basename(input_video)} ({width}x{height})")
    print(f"📐 Output video: {width}x{height} (ORIGINAL DIMENSIONS)")
    print(f"📐 Crop windows: 2x 640x640 (second with smart black padding)")
    print(f"🖼️  Frames processed: {frame_count}/{total_frames}")
    print(f"🎯 TRACKING RESULTS:")
    print(f"   📊 Total vehicles tracked: {total_tracked_final}")
    print(f"   ✅ Confirmed license plates: {confirmed_final}")
    print(f"   🎯 Tracking threshold: {tracking_threshold}px")
    print(f"📹 Output video: {output_video}")
    
    # Print detected plate texts
    if tracked_plates:
        print(f"\n📋 Detected License Plates:")
        print(f"{'='*60}")
        for plate_id, plate_info in tracked_plates.items():
            text = plate_info.get('best_text', 'No text detected')
            confidence = int(plate_info.get('best_confidence', 0))
            frames = plate_info.get('frame_count', 0)
            if text and text != 'No text detected':
                print(f"ID {plate_info['id']}: {text} (confidence: {confidence}%, {frames} frames)")
        print(f"{'='*60}")
    
    print(f"\n🎨 Visual Elements:")
    print(f"   🟢 Green: Crop 1 window (normal extraction)")
    print(f"   🔴 Red: Crop 2 window (with padding info)")
    print(f"   🎯 Bounding boxes: Only drawn on original video area")
    print(f"   ⚫ Black padding: Used internally for detection only")
    print(f"\n🎯 Process completed successfully!")

if __name__ == "__main__":
    main()