"""
Video License Plate Detection System
Performs ALPR on video files with session-based statistics tracking
"""

import sys
import os
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict
import warnings

import cv2
import numpy as np
import torch

# Suppress only specific ONNX Runtime warnings
warnings.filterwarnings('ignore', message='.*TensorRT.*')

from fast_alpr import ALPR
from fast_alpr.base import BaseDetector, DetectionResult

# Try to load configuration from config.py
try:
    import config
    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False

# Configuration constants
if CONFIG_LOADED:
    YOLO_CONFIDENCE_THRESHOLD = getattr(config, 'YOLO_CONFIDENCE_THRESHOLD', 0.5)
    YOLO_IOU_THRESHOLD = getattr(config, 'YOLO_IOU_THRESHOLD', 0.4)
    MAX_DISPLAY_WIDTH = getattr(config, 'MAX_DISPLAY_WIDTH', 1690)
    MAX_DISPLAY_HEIGHT = getattr(config, 'MAX_DISPLAY_HEIGHT', 923)
    DEFAULT_MODEL_PATH = getattr(config, 'MODEL_PATH', r'D:\EPIC\plate_detector\utils\models\best.pt')
    DEFAULT_OCR_MODEL = getattr(config, 'OCR_MODEL', 'cct-s-v1-global-model')
    print("[OK] Configuration loaded from config.py")
else:
    YOLO_CONFIDENCE_THRESHOLD = 0.5
    YOLO_IOU_THRESHOLD = 0.4
    MAX_DISPLAY_WIDTH = 1690
    MAX_DISPLAY_HEIGHT = 923
    DEFAULT_MODEL_PATH = os.getenv('MODEL_PATH', r'D:\EPIC\plate_detector\utils\models\best.pt')
    DEFAULT_OCR_MODEL = os.getenv('OCR_MODEL', 'cct-s-v1-global-model')
    print("[WARNING] config.py not found, using default configuration")

# Additional constants
CROP_CORNER_SIZE = 15
VISUALIZATION_CORNER_SIZE = 25
VISUALIZATION_ALPHA = 0.7
FPS_DEFAULT = 25.0

# GPU configuration
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


@dataclass
class PlateStatistics:
    count: int = 0
    last_printed: bool = False

class PlateSessionTracker:
    def __init__(self):
        self.statistics: Dict[str, PlateStatistics] = defaultdict(PlateStatistics)
        self.total_detections: int = 0
        self.frames_with_detections: int = 0
    
    def add_detection(self, plate_text: str) -> None:
        self.statistics[plate_text].count += 1
        self.total_detections += 1
    
    def increment_frame_count(self) -> None:
        """Increment the frame counter"""
        self.frames_with_detections += 1
    
    def get_average_per_plate(self) -> float:
        """Calculate average detections per unique plate"""
        num_unique = len(self.statistics)
        if num_unique == 0:
            return 0.0
        return self.total_detections / num_unique
    
    def get_plates_above_average(self) -> List[Tuple[str, PlateStatistics]]:
        """Get plates that appear more than average"""
        avg = self.get_average_per_plate()
        return [
            (text, stats) 
            for text, stats in self.statistics.items()
            if stats.count >= avg and not stats.last_printed
        ]
    
    def mark_as_printed(self, plate_text: str) -> None:
        """Mark a plate as printed"""
        if plate_text in self.statistics:
            self.statistics[plate_text].last_printed = True
    
    def reset(self) -> None:
        """Reset all statistics"""
        self.statistics.clear()
        self.total_detections = 0
        self.frames_with_detections = 0
    
    def __len__(self) -> int:
        """Return number of unique plates"""
        return len(self.statistics)


# Check for required dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[ERROR] Ultralytics not available")
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
                print(f"[OK] Custom YOLO detector loaded from: {model_path}")
                print(f"[GPU] Running on GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print(f"[OK] Custom YOLO detector loaded from: {model_path}")
                print("[WARNING] GPU not available, running on CPU")
                
        except Exception as e:
            print(f"[ERROR] Failed to load custom YOLO model: {e}")
            raise

    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        # Run inference on GPU with optimized settings
        results = self.model(
            frame, 
            conf=YOLO_CONFIDENCE_THRESHOLD, 
            iou=YOLO_IOU_THRESHOLD, 
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

def create_alpr_pipeline(model_path: str, ocr_model: str = DEFAULT_OCR_MODEL) -> ALPR:
    """
    Create and initialize ALPR pipeline
    
    Args:
        model_path: Path to YOLO model weights
        ocr_model: OCR model name to use
    
    Returns:
        Initialized ALPR instance
    """
    try:
        print("[LOADING] Attempting to load custom detector model...")
        custom_detector = CustomYOLODetector(model_path)
        
        alpr = ALPR(
            detector=custom_detector, 
            ocr_model=ocr_model
        )
        print("[OK] Custom detector model loaded successfully!")
        print("[OCR] OCR will use best available device (GPU preferred)")
        return alpr
    except Exception as e:
        print(f"[ERROR] Failed to load custom detector model: {e}")
        print("[EXIT] Exiting program...")
        sys.exit(1)


def validate_vietnamese_plate_format(text: str) -> bool:
    """Validate Vietnamese license plate format"""
    if not text:
        return False
    
    clean_text = text.replace(' ', '').replace('-', '').replace('.', '').upper()
    
    if len(clean_text) < 8 or len(clean_text) > 9:
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

def format_plate(text: str) -> str:
    """Format Vietnamese license plate text"""
    if not text:
        return ""
    
    clean_text = text.replace(' ', '').replace('-', '').replace('.', '').upper()
    
    if len(clean_text) == 8 or len(clean_text) == 9:
        first_three = clean_text[:3]  
        last = clean_text[3:]     
        return f"{first_three}-{last}"
    return clean_text

def extract_crop(frame: np.ndarray, crop_x: int, crop_y: int, crop_size: int, crop_index: int) -> Optional[np.ndarray]:
    """
    Extract a crop window from the frame with padding if needed
    
    Args:
        frame: Source frame
        crop_x: Row position
        crop_y: Column position
        crop_size: Size of crop window
        crop_index: Index of crop (0 = first, 1+ = with padding)
    
    Returns:
        Cropped frame or None if invalid
    """
    if crop_index == 0:
        # First crop - normal extraction
        if crop_y + crop_size <= frame.shape[1] and crop_x + crop_size <= frame.shape[0]:
            return frame[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size].copy()
        else:
            return None
    else:
        # Additional crops - with black padding if needed
        available_width = frame.shape[1] - crop_y
        available_height = frame.shape[0] - crop_x
        
        if available_width <= 0 or available_height <= 0:
            return None
        
        # Create crop with black padding
        crop_frame = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        
        # Extract what we can from the original frame
        actual_width = min(available_width, crop_size)
        actual_height = min(available_height, crop_size)
        
        if actual_width > 0 and actual_height > 0:
            actual_crop = frame[crop_x:crop_x + actual_height, crop_y:crop_y + actual_width]
            crop_frame[:actual_height, :actual_width] = actual_crop
            return crop_frame
        
        return None


def run_detection_on_crops(frame: np.ndarray, alpr: ALPR, x: int, y: int, times: int, crop_size: int) -> Tuple[List[List[int]], Dict[str, Dict]]:
    """
    Run ALPR detection on crop windows
    
    Args:
        frame: Source frame
        alpr: ALPR pipeline instance
        x: Row position for crops
        y: Column position for crops
        times: Number of crop windows
        crop_size: Size of each crop window
    
    Returns:
        Tuple of (detections, alpr_data)
    """
    detections = []
    alpr_data = {}
    
    for i in range(times):
        crop_x = x
        crop_y = y + crop_size * i
        
        # Extract crop
        crop_frame = extract_crop(frame, crop_x, crop_y, crop_size, i)
        if crop_frame is None:
            continue
        
        # Run ALPR detection
        alpr_results = alpr.predict(crop_frame)
        
        if alpr_results:
            for result in (alpr_results if isinstance(alpr_results, list) else [alpr_results]):
                if hasattr(result, 'detection') and hasattr(result.detection, 'bounding_box'):
                    bbox = result.detection.bounding_box
                    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                    
                    # Convert to original frame coordinates
                    orig_x1 = crop_y + x1
                    orig_y1 = crop_x + y1
                    orig_x2 = crop_y + x2
                    orig_y2 = crop_x + y2
                    
                    # Validate boundaries
                    if (orig_x1 < frame.shape[1] and orig_y1 < frame.shape[0] and 
                        orig_x2 < frame.shape[1] and orig_y2 < frame.shape[0] and
                        orig_x1 >= 0 and orig_y1 >= 0):
                        
                        text = result.ocr.text if hasattr(result, 'ocr') and result.ocr.text else ""
                        confidence = result.ocr.confidence if hasattr(result, 'ocr') and result.ocr.confidence else 0.0
                        
                        detection = [orig_x1, orig_y1, orig_x2, orig_y2]
                        detections.append(detection)
                        alpr_data[str(detection)] = {'text': text, 'confidence': confidence}
    
    return detections, alpr_data


def draw_detection_box(frame: np.ndarray, detection: List[int], text: str, is_valid: bool) -> None:
    """
    Draw bounding box and text for a detection
    
    Args:
        frame: Frame to draw on (modified in place)
        detection: Bounding box [x1, y1, x2, y2]
        text: Plate text to display
        is_valid: Whether the plate format is valid
    """
    x1, y1, x2, y2 = detection
    
    # Choose color and text
    if is_valid:
        color = (0, 255, 0)  # Green for valid plates
        display_text = text
    else:
        color = (0, 255, 255)  # Yellow for reading/invalid
        display_text = "Reading..."
    
    # Draw main rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Draw corner markers
    corner_size = CROP_CORNER_SIZE
    corner_thickness = 3
    
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_size, y1), color, corner_thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_size), color, corner_thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_size, y1), color, corner_thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_size), color, corner_thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_size, y2), color, corner_thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_size), color, corner_thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_size, y2), color, corner_thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_size), color, corner_thickness)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    text_y = y1 - 10 if y1 > 40 else y2 + 30
    cv2.putText(frame, display_text, (x1, text_y), font, font_scale, color, thickness)


def print_plate_texts(tracker: PlateSessionTracker) -> List[str]:
    """
    Print plate texts that meet the threshold criteria
    
    Args:
        tracker: Session tracker with plate statistics
    
    Returns:
        List of printed plate texts
    """
    printed_plates = []
    
    if len(tracker) == 0:
        return printed_plates
    
    try:
        # Get plates above average
        for text, stats in tracker.get_plates_above_average():
            print(f"[DETECTED PLATE] {text} (Count: {stats.count})")
            printed_plates.append(text)
            tracker.mark_as_printed(text)
    
    except Exception as e:
        print(f"Error printing plate texts: {e}")
    
    return printed_plates


def draw_crop_windows(frame: np.ndarray, x: int, y: int, times: int, crop_size: int) -> np.ndarray:
    """
    Draw crop windows on frame for visualization
    
    Args:
        frame: Input frame
        x: Row position
        y: Column position
        times: Number of crop windows
        crop_size: Size of each window
    
    Returns:
        Frame with crop windows drawn
    """
    overlay = frame.copy()
    
    for i in range(times):
        crop_x = x
        crop_y = y + crop_size * i
        
        # Calculate visible dimensions
        display_width = min(crop_size, frame.shape[1] - crop_y)
        display_height = min(crop_size, frame.shape[0] - crop_x)
        
        if display_width <= 0 or display_height <= 0:
            continue
        
        # Choose color
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        color = colors[i % len(colors)]
        
        # Draw main rectangle
        cv2.rectangle(overlay, (crop_y, crop_x), 
                     (crop_y + display_width, crop_x + display_height), color, 3)
        
        # Draw corner markers
        corner_size = VISUALIZATION_CORNER_SIZE
        corner_thickness = 5
        
        # Top-left
        cv2.line(overlay, (crop_y, crop_x), 
                (crop_y + min(corner_size, display_width), crop_x), color, corner_thickness)
        cv2.line(overlay, (crop_y, crop_x), 
                (crop_y, crop_x + min(corner_size, display_height)), color, corner_thickness)
        
        # Top-right
        cv2.line(overlay, (crop_y + display_width, crop_x), 
                (crop_y + display_width - min(corner_size, display_width), crop_x), color, corner_thickness)
        cv2.line(overlay, (crop_y + display_width, crop_x), 
                (crop_y + display_width, crop_x + min(corner_size, display_height)), color, corner_thickness)
        
        # Bottom-left
        cv2.line(overlay, (crop_y, crop_x + display_height), 
                (crop_y + min(corner_size, display_width), crop_x + display_height), color, corner_thickness)
        cv2.line(overlay, (crop_y, crop_x + display_height), 
                (crop_y, crop_x + display_height - min(corner_size, display_height)), color, corner_thickness)
        
        # Bottom-right
        cv2.line(overlay, (crop_y + display_width, crop_x + display_height), 
                (crop_y + display_width - min(corner_size, display_width), crop_x + display_height), color, corner_thickness)
        cv2.line(overlay, (crop_y + display_width, crop_x + display_height), 
                (crop_y + display_width, crop_x + display_height - min(corner_size, display_height)), color, corner_thickness)
    
    # Blend overlay
    result = cv2.addWeighted(frame, VISUALIZATION_ALPHA, overlay, 1 - VISUALIZATION_ALPHA, 0)
    return result

def process_frame_with_crops(frame: np.ndarray, alpr: ALPR, tracker: PlateSessionTracker, x: int, y: int, times: int, 
                            crop_size: int, show_crop_windows: bool = True) -> Tuple[np.ndarray, int, int, List[str]]:
    """
    Process frame with crop detection and proportional-based printing
    
    Args:
        frame: Input frame
        alpr: ALPR pipeline instance
        tracker: Plate session tracker
        x: Row position for crops
        y: Column position for crops
        times: Number of crop windows
        crop_size: Size of each crop window
        show_crop_windows: Whether to visualize crop windows
    
    Returns:
        Tuple of (processed_frame, num_detections, num_valid, printed_plates)
    """
    result_frame = frame.copy()
    
    # Draw crop windows visualization
    if show_crop_windows:
        result_frame = draw_crop_windows(result_frame, x, y, times, crop_size)
    
    # Run detection on all crops
    detections, alpr_data = run_detection_on_crops(frame, alpr, x, y, times, crop_size)
    
    # Process detections and collect valid plates
    all_detected_texts = []
    
    for detection in detections:
        x1, y1, x2, y2 = detection
        detection_key = str([x1, y1, x2, y2])
        
        text = ""
        is_valid = False
        
        if detection_key in alpr_data:
            text = alpr_data[detection_key]['text']
            if text and validate_vietnamese_plate_format(text):
                formatted_text = format_plate(text)
                all_detected_texts.append(formatted_text)
                is_valid = True
                
                # Draw detection box
                draw_detection_box(result_frame, detection, formatted_text, is_valid)
            else:
                draw_detection_box(result_frame, detection, text, is_valid)
        else:
            draw_detection_box(result_frame, detection, "", is_valid)
    
    # Session management
    printed_plates_this_frame = []
    
    if len(all_detected_texts) == 0:
        # No plates detected - print eligible plates and reset
        if len(tracker) > 0:
            num_unique_plates = len(tracker)
            if num_unique_plates > 0 and tracker.total_detections > 0:
                avg_per_plate = tracker.total_detections / num_unique_plates
                print(f"[SESSION END] {tracker.frames_with_detections} frames, "
                      f"{tracker.total_detections} total, {num_unique_plates} unique")
                print(f"[AVERAGE] {avg_per_plate:.1f} detections per plate")
                
                # Print qualifying plates
                printed_plates_this_frame = print_plate_texts(tracker)
            
            # Reset session
            tracker.reset()
    else:
        # Plates detected - update statistics
        tracker.increment_frame_count()
        for text in all_detected_texts:
            tracker.add_detection(text)
    
    # Return results
    detected_count = len(detections)
    valid_count = len(all_detected_texts)
    return result_frame, detected_count, valid_count, printed_plates_this_frame

def main():
    """Main function to process video with detection and save output"""
    # Check for model file
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"[ERROR] Model file not found: {DEFAULT_MODEL_PATH}")
        sys.exit(1)
    
    # Get input and output paths from command line or use defaults
    if len(sys.argv) > 1:
        input_video_path = sys.argv[1]
    else:
        input_video_path = input("Enter input video path: ").strip('"')
    
    if not os.path.exists(input_video_path):
        print(f"[ERROR] Input video not found: {input_video_path}")
        sys.exit(1)
    
    if len(sys.argv) > 2:
        output_video_path = sys.argv[2]
    else:
        # Create default output path in D:\EPIC\plate_detector\video\output
        output_dir = r"D:\EPIC\plate_detector\video\output"
        os.makedirs(output_dir, exist_ok=True)
        input_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path = os.path.join(output_dir, f"{input_name}_processed.mp4")
    
    print(f"[INPUT] Video: {input_video_path}")
    print(f"[OUTPUT] Video: {output_video_path}")
    print("=" * 70)
    
    # Create ALPR pipeline
    alpr = create_alpr_pipeline(DEFAULT_MODEL_PATH, DEFAULT_OCR_MODEL)
    
    # Create plate tracker
    tracker = PlateSessionTracker()
    
    # Crop parameters - use config.py if available
    if CONFIG_LOADED:
        x = getattr(config, 'CROP_X', 100)
        y = getattr(config, 'CROP_Y', 400)
        times = getattr(config, 'CROP_TIMES', 1)
        crop_size = getattr(config, 'CROP_SIZE', 640)
        show_crop_windows = getattr(config, 'SHOW_CROP_WINDOWS', True)
    else:
        x = 100  # row position
        y = 400  # column position
        times = 1  # number of crop windows
        crop_size = 640  # size of each crop window
        show_crop_windows = True
    
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {input_video_path}")
        sys.exit(1)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = FPS_DEFAULT
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[VIDEO] Resolution: {width}x{height}")
    print(f"[VIDEO] FPS: {fps:.2f}")
    print(f"[VIDEO] Total frames: {total_frames}")
    print(f"[CONFIG] Crop parameters: x={x}, y={y}, windows={times}, size={crop_size}")
    print("=" * 70)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"[ERROR] Failed to create output video: {output_video_path}")
        cap.release()
        sys.exit(1)
    
    frame_count = 0
    total_printed = 0
    start_time = time.time()
    
    try:
        print("[START] Processing video...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame with crop detection and tracking
            processed_frame, active_objects, confirmed_plates, printed_plates = process_frame_with_crops(
                frame, alpr, tracker, x, y, times, crop_size, show_crop_windows=show_crop_windows
            )
            
            # Count printed plates
            if printed_plates:
                total_printed += len(printed_plates)
            
            # Write processed frame to output video
            out.write(processed_frame)
            
            # Print progress every 1000 frames
            if frame_count % 1000 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"[PROGRESS] Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                      f"Processing: {fps_processing:.1f} fps - "
                      f"Detected: {active_objects}, Valid: {confirmed_plates}")
    
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
    finally:
        # Clean up
        cap.release()
        out.release()
        
        elapsed_time = time.time() - start_time
        
        print("=" * 70)
        print("[SUMMARY] PROCESSING SUMMARY")
        print("=" * 70)
        print(f"   Frames processed: {frame_count}/{total_frames}")
        print(f"   Processing time: {elapsed_time:.2f} seconds")
        print(f"   Average FPS: {frame_count / elapsed_time:.2f}")
        print(f"   Unique plates detected: {total_printed}")
        print(f"   Output saved to: {output_video_path}")
        print("[DONE] Processing complete!")

if __name__ == "__main__":
    main()
