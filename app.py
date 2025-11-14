"""
Plate Detection Web Server Application
Flask-based web server for license plate detection and monitoring
"""

import os
import sys

# Suppress FFmpeg warnings about missing codecs before importing cv2
os.environ['FFREPORT'] = 'level=quiet'

import cv2
import json
import base64
import logging
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

# Add utils directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

# Import detection modules from lp_detection_video
from lp_detection_video import (
    PlateStatistics,
    PlateSessionTracker,
    CustomYOLODetector,
    create_alpr_pipeline,
    validate_vietnamese_plate_format,
    format_plate,
    extract_crop,
    run_detection_on_crops,
    draw_detection_box,
    draw_crop_windows,
    process_frame_with_crops,
    DEFAULT_MODEL_PATH,
    DEFAULT_OCR_MODEL
)

# Import utils modules
from utils import plate_deduplication
from utils import plate_persistence
from utils.camera_manager import camera_manager
from utils.constants import (
    FPS_TARGET, FRAME_DELAY, JPEG_QUALITY, GC_INTERVAL_FRAMES
)

# Session trackers for each camera
camera_trackers = {}

# Global ALPR instance
alpr = None
alpr_lock = threading.Lock()

# ============================================================================
# PLATE TRACKING FUNCTIONS 
# ============================================================================

# Global tracking variables for video processing
tracked_plates = {}
plate_id_counter = 0
tracking_threshold = 50  # Distance threshold for tracking in pixels

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
        if clean_length >= 8 and clean_length <= 9:
            plate_info['text_history'].append(formatted_text)

            # Keep only last 10 detections for optimal performance and accuracy
            if len(plate_info['text_history']) > 30:
                plate_info['text_history'] = plate_info['text_history'][-30:]

            # Recalculate best text from entire history (not just new text)
            if plate_info['text_history']:
                # Count frequency of each unique text
                from collections import Counter
                text_counts = Counter(plate_info['text_history'])
                
                # Get the most common text
                most_common_text, count = text_counts.most_common(1)[0]
                current_confidence = (count / len(plate_info['text_history'])) * 100
                
                # Always use the most frequent text
                plate_info['best_text'] = most_common_text
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
    """Update tracking using distance-based matching"""
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
            # Remove plates not seen for more than 30 frames
            if tracked_plates[plate_id]['frames_since_seen'] > 30:
                plates_to_remove.append(plate_id)
    
    # Remove old plates
    for plate_id in plates_to_remove:
        del tracked_plates[plate_id]
    
    return current_plates

def reset_tracking():
    """Reset all tracking state"""
    global tracked_plates, plate_id_counter
    tracked_plates = {}
    plate_id_counter = 0

# ============================================================================
# END OF TRACKING FUNCTIONS
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO for better performance
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(32).hex())
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size for videos

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state
video_streams = {}  # Store active video stream flags
models_initialized = False

# Thread pool for async operations (keep from remote)
persistence_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="persist")
cleanup_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cleanup")


def init_alpr_models():
    """Initialize ALPR models"""
    global alpr
    with alpr_lock:
        if alpr is None:
            try:
                alpr = create_alpr_pipeline(DEFAULT_MODEL_PATH, DEFAULT_OCR_MODEL)
                logger.info("ALPR models initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize ALPR models: {e}")
                return False
        return True


def get_crop_config():
    """Get crop configuration from config.py or use defaults"""
    try:
        import config
        return {
            'x': getattr(config, 'CROP_X', 100),
            'y': getattr(config, 'CROP_Y', 400),
            'times': getattr(config, 'CROP_TIMES', 1),
            'crop_size': getattr(config, 'CROP_SIZE', 640),
            'show_crop_windows': getattr(config, 'SHOW_CROP_WINDOWS', True)
        }
    except ImportError:
        return {
            'x': 100,
            'y': 400,
            'times': 1,
            'crop_size': 640,
            'show_crop_windows': True
        }


def encode_image_base64(image):
    """Encode OpenCV image to base64 string"""
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        logger.error("Failed to encode image to JPEG")
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode('utf-8')


def get_camera_stream(camera_id, camera_source=0):
    """
    Generator function for camera streaming with session-based plate tracking
    
    Args:
        camera_id: Camera identifier (1 or 2)
        camera_source: Camera source (0 for default, or URL for IP camera)
    """
    import gc
    
    # Get or initialize camera using CameraManager
    camera = camera_manager.get_camera(camera_id)
    if camera is None:
        camera = camera_manager.add_camera(camera_id, camera_source)
        if camera is None:
            logger.error("Failed to initialize camera %s", camera_id)
            return
    
    # Initialize session tracker for this camera if not exists
    if camera_id not in camera_trackers:
        camera_trackers[camera_id] = PlateSessionTracker()
    
    tracker = camera_trackers[camera_id]
    frame_count = 0
    
    while camera_manager.is_active(camera_id):
        try:
            # Acquire camera lock and read frame
            with camera_manager.acquire_camera(camera_id) as cap:
                if cap is None:
                    logger.warning("Camera %s not available", camera_id)
                    break
                
                success, frame = cap.read()
            
            if not success:
                logger.warning("Failed to read frame from camera %s", camera_id)
                break
            
            frame_count += 1
            
            # Get crop configuration
            crop_config = get_crop_config()
            
            # Process frame with plate detection using lp_detection_video logic
            processed_frame, detected_count, valid_count, _ = process_frame_with_crops(
                frame, alpr, tracker, 
                crop_config['x'], crop_config['y'], 
                crop_config['times'], crop_config['crop_size'],
                show_crop_windows=crop_config['show_crop_windows']
            )
            
            # Extract valid plate numbers from session tracker
            all_detected_texts = []
            detected_plates = []
            for plate_text, stats in tracker.statistics.items():
                if stats.count > 0 and not stats.last_printed:
                    all_detected_texts.append(plate_text)
                    detected_plates.append({
                        'plate_number': plate_text,
                        'confidence': 100,
                        # 'bbox': [0, 0, 100, 100],  # Placeholder
                        'plate_id': hash(plate_text) % 10000
                    })
            
            # Session management (like lp_detection_video.py)
            if len(all_detected_texts) == 0:
                # No plates detected - check if we should print qualifying plates and reset
                if len(tracker) > 0:
                    num_unique_plates = len(tracker)
                    if num_unique_plates > 0 and tracker.total_detections > 0:
                        avg_per_plate = tracker.total_detections / num_unique_plates
                        logger.info(f"[Cam {camera_id}] Session end - {tracker.frames_with_detections} frames, "
                                  f"{tracker.total_detections} total, {num_unique_plates} unique, "
                                  f"avg: {avg_per_plate:.1f}")
                        
                        # Print plates above average
                        for text, stats in tracker.get_plates_above_average():
                            if not stats.last_printed:
                                logger.info(f"[Cam {camera_id}] DETECTED PLATE: {text} (Count: {stats.count})")
                                tracker.mark_as_printed(text)
                    
                    # Reset session
                    tracker.reset()
            else:
                # Plates detected - update statistics
                tracker.increment_frame_count()
                for text in all_detected_texts:
                    tracker.add_detection(text)
            
            # Save detections to persistence layer (async to avoid blocking)
            if detected_plates:
                for plate_info in detected_plates:
                    # Check if should save (deduplication)
                    plate_number = plate_info['plate_number']
                    confidence = plate_info['confidence']
                    
                    # Determine direction based on camera
                    direction = 'IN' if camera_id == 1 else 'OUT'
                    
                    # Check deduplication
                    x1, y1, x2, y2 = plate_info['bbox']
                    plate_crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                    
                    is_dup, _reason = plate_deduplication.is_duplicate(
                        camera_id, plate_number, plate_crop, confidence, plate_info['bbox']
                    )
                    
                    if not is_dup:
                        # Check if should save image
                        should_save, _save_reason = plate_deduplication.should_save_image(
                            camera_id, plate_info['plate_id'], plate_number, plate_info['bbox']
                        )
                        
                        if should_save:
                            logger.info("Saving new detection: %s (Camera %s)", plate_number, camera_id)
                            # Persist detection asynchronously to avoid blocking stream
                            frame_copy = frame.copy()  # Copy frame for async processing
                            persistence_executor.submit(
                                plate_persistence.persist_plate_detection,
                                frame=frame_copy,
                                camera_id=camera_id,
                                plate_number=plate_number,
                                confidence=confidence,
                                direction=direction,
                                tracking_id=plate_info['plate_id'],
                                is_duplicate=False
                            )
            
            # Encode frame to JPEG with optimized quality
            success, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not success:
                logger.error("Failed to encode frame to JPEG")
                continue
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Garbage collection every N frames to prevent memory leaks
            if frame_count % GC_INTERVAL_FRAMES == 0:
                gc.collect()
            
            # Control frame rate
            time.sleep(FRAME_DELAY)
            
        except Exception as e:
            logger.error("Error in camera stream %s: %s", camera_id, str(e))
            import traceback
            logger.error(traceback.format_exc())
            break
    
    # Cleanup
    logger.info("Stopping camera %s stream", camera_id)


def release_camera(camera_id):
    """Release camera resource"""
    camera_manager.remove_camera(camera_id)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/test')
def test_page():
    """API test page"""
    return render_template('test.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    global models_initialized, alpr
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': alpr is not None,
        'models_initialized': models_initialized,
        'active_cameras': camera_manager.get_active_cameras()
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        config_data = get_crop_config()
        return jsonify({
            'success': True,
            'config': config_data
        })
    except (IOError, KeyError, ValueError) as e:
        logger.error("Error getting config: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        new_config = request.json
        
        if not new_config:
            return jsonify({
                'success': False,
                'error': 'No configuration data provided'
            }), 400
        
        # Update config.py file directly
        import config
        for key, value in new_config.items():
            setattr(config, key, value)
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully'
        })
    except (IOError, ValueError, TypeError) as e:
        logger.error("Error updating config: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/init-models', methods=['POST'])
def init_models():
    """Initialize detection models"""
    global models_initialized
    try:
        success = init_alpr_models()
        models_initialized = success
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Models initialized successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to initialize models - check model files exist'
            }), 500
    except (IOError, RuntimeError, ValueError) as e:
        logger.error("Error initializing models: %s", str(e))
        models_initialized = False
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded/processed video files"""
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    # Set proper MIME type for video files
    if filename.endswith('.avi'):
        response.headers['Content-Type'] = 'video/x-msvideo'
    elif filename.endswith('.mp4'):
        response.headers['Content-Type'] = 'video/mp4'
    elif filename.endswith('.mkv'):
        response.headers['Content-Type'] = 'video/x-matroska'
    # Enable range requests for video streaming
    response.headers['Accept-Ranges'] = 'bytes'
    return response


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera stream"""
    global models_initialized
    try:
        data = request.json or {}
        camera_id = data.get('camera_id', 1)
        camera_source = data.get('camera_source', 0)
        
        # Validate camera_id
        if not isinstance(camera_id, int) or camera_id not in [1, 2]:
            return jsonify({
                'success': False,
                'error': 'Invalid camera_id. Must be 1 or 2'
            }), 400
        
        # Initialize models if needed
        if not models_initialized or alpr is None:
            logger.info("Initializing models for camera stream...")
            models_initialized = init_alpr_models()
            if not models_initialized:
                return jsonify({
                    'success': False,
                    'error': 'Failed to initialize detection models'
                }), 500
        
        # Mark camera as active
        video_streams[camera_id] = True
        
        return jsonify({
            'success': True,
            'message': f'Camera {camera_id} stream started',
            'camera_id': camera_id,
            'stream_url': f'/api/camera/stream/{camera_id}'
        })
    except (IOError, RuntimeError, ValueError) as e:
        logger.error("Error starting camera: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera stream"""
    try:
        data = request.json or {}
        camera_id = data.get('camera_id', 1)
        
        # Release camera
        release_camera(camera_id)
        
        # Remove from active streams
        if camera_id in video_streams:
            del video_streams[camera_id]
        
        return jsonify({
            'success': True,
            'message': f'Camera {camera_id} stream stopped'
        })
    except (RuntimeError, ValueError) as e:
        logger.error("Error stopping camera: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/camera/stream/<int:camera_id>')
def camera_stream(camera_id):
    """Video streaming route"""
    try:
        camera_source = request.args.get('source', 0)
        
        # Convert source to int if it's a number
        try:
            camera_source = int(camera_source)
        except ValueError:
            pass  # It's a string URL, keep it as is
        
        return Response(
            get_camera_stream(camera_id, camera_source),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except (ValueError, TypeError) as e:
        logger.error("Error in camera stream: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """Get camera stream status"""
    try:
        camera_id = request.args.get('camera_id', type=int)
        
        if camera_id:
            is_active = camera_manager.is_active(camera_id)
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'is_active': is_active
            })
        else:
            # Return status for all cameras
            status = {
                cam_id: camera_manager.is_active(cam_id)
                for cam_id in [1, 2]
            }
            return jsonify({
                'success': True,
                'cameras': status,
                'active_cameras': camera_manager.get_active_cameras()
            })
    except (RuntimeError, ValueError) as e:
        logger.error("Error getting camera status: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Process uploaded video file for plate detection"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        allowed_video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if ext not in allowed_video_extensions:
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(allowed_video_extensions)}'
            }), 400
        
        # Get camera_id from form data
        try:
            camera_id = int(request.form.get('camera_id', 1))
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid camera_id'
            }), 400
        
        # Save video temporarily
        temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_video_path)
        
        # Initialize models if needed
        if alpr is None:
            logger.info("Initializing models for video processing...")
            if not init_alpr_models():
                os.remove(temp_video_path)
                return jsonify({
                    'success': False,
                    'error': 'Failed to initialize models'
                }), 500
        
        # Process video
        start_time = time.time()
        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            os.remove(temp_video_path)
            return jsonify({
                'success': False,
                'error': 'Failed to open video file'
            }), 400
        
        all_detections = []
        frames_processed = 0
        frame_skip = 3  # Process every 3rd frame (changed from 5)
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Starting video processing for camera {camera_id}")
        logger.info(f"Video resolution: {video_width}x{video_height}")
        logger.info(f"Video FPS: {video_fps}")
        logger.info(f"Total frames: {total_frames}")
        
        # Create output video with detections
        # Use MJPEG for Windows (most reliable), fallback to other codecs
        base_name = os.path.splitext(secure_filename(file.filename))[0]
        
        # Try different codecs in order of preference
        codecs_to_try = [
            ('avc1', '.mp4'),  # H.264 - best quality and compatibility
            ('mp4v', '.mp4'),  # MPEG-4 - good compatibility
            ('XVID', '.avi'),  # Xvid - good compression
            ('MJPG', '.avi'),  # MJPEG - always works but large files
        ]
        
        # Suppress FFmpeg codec loading warnings
        import logging as cv_logging
        cv_logger = cv_logging.getLogger('ffmpeg')
        cv_logger.setLevel(cv_logging.CRITICAL)
        
        out = None
        codec_used = None
        output_video_path = None
        output_video_name = None
        
        for codec, ext in codecs_to_try:
            try:
                output_video_name = f"processed_{base_name}{ext}"
                output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], output_video_name)
                
                # Remove existing file if it exists
                if os.path.exists(output_video_path):
                    try:
                        os.remove(output_video_path)
                    except:
                        pass
                
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (video_width, video_height))
                
                if test_out.isOpened():
                    logger.info(f"✓ Using video codec: {codec} ({ext})")
                    out = test_out
                    codec_used = codec
                    break
                else:
                    test_out.release()
                    if os.path.exists(output_video_path):
                        try:
                            os.remove(output_video_path)
                        except:
                            pass
                    logger.debug(f"Codec {codec} not available, trying next...")
            except Exception as e:
                logger.debug(f"Error trying codec {codec}: {str(e)[:100]}")
                if os.path.exists(output_video_path):
                    try:
                        os.remove(output_video_path)
                    except:
                        pass
        
        if out is None or not out.isOpened():
            cap.release()
            os.remove(temp_video_path)
            logger.error("Failed to initialize video writer with ANY codec. Check OpenCV/FFmpeg installation.")
            return jsonify({
                'success': False,
                'error': 'Video writer initialization failed. Please check FFmpeg/codec installation.'
            }), 500
        
        # Store detection info per frame
        frame_detections = {}
        
        # Reset tracking for this video
        reset_tracking()
        
        # Dictionary to store all confirmed plates (won't be removed)
        confirmed_plates = {}
        
        # Reset video capture to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames_processed = 0
        
        logger.info(f"Starting frame-by-frame processing with plate tracking...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_processed += 1
            
            # Get crop config
            crop_config = get_crop_config()
            
            # Process frame with detection and tracking
            try:
                # Draw crop windows first
                display_frame = frame.copy()
                display_frame = draw_crop_windows(
                    display_frame,
                    crop_config['x'], crop_config['y'],
                    crop_config['times'], crop_config['crop_size']
                )
                
                # Run detection on crops to get bounding boxes
                detections, alpr_data = run_detection_on_crops(
                    frame, alpr,
                    crop_config['x'], crop_config['y'],
                    crop_config['times'], crop_config['crop_size']
                )
                
                # Update tracking with new detections
                current_plates = update_tracking(detections)
                
                # Process each tracked plate
                for plate_id, plate_info in current_plates.items():
                    x1, y1, x2, y2 = plate_info['bbox']
                    
                    # Get ALPR text for this detection
                    detection_key = str([x1, y1, x2, y2])
                    if detection_key in alpr_data:
                        recognized_text = alpr_data[detection_key]['text']
                        if recognized_text:
                            # Update text history and get best text
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
                        elif confidence_percentage > 40:
                            color = (0, 255, 255)  # Yellow for medium confidence
                        else:
                            color = (255, 255, 0)  # Cyan for low confidence
                    else:
                        color = (0, 255, 255)  # Yellow for reading
                        display_text = "Reading..."
                        confidence_percentage = 0
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 4)
                    
                    # Draw tracking ID and plate text
                    label = f"ID:{plate_info['id']} {display_text}"
                    # confidence_label = f"{confidence_percentage}% ({frequency}/{total_detections})"
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2
                    
                    # Get text size
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Position text above bounding box
                    text_x = x1
                    text_y = y1 - 10 if y1 > 30 else y2 + text_height + 10
                    
                    # Draw text
                    cv2.putText(display_frame, label, (text_x, text_y), 
                               font, font_scale, color, thickness)
                    
                    # Draw confidence info
                    conf_y = text_y + 25 if text_y > y1 else text_y + text_height + 25
                    # # cv2.putText(display_frame, confidence_label, (text_x, conf_y), 
                    #            font, font_scale * 0.8, color, thickness)
                        
            except Exception as e:
                logger.warning(f"Error processing frame {frames_processed}: {e}")
                # Fallback to just drawing crop windows
                display_frame = frame.copy()
                display_frame = draw_crop_windows(
                    display_frame,
                    crop_config['x'], crop_config['y'],
                    crop_config['times'], crop_config['crop_size']
                )
            
            # Write frame to output video
            out.write(display_frame)
            
            # Save confirmed plates to permanent storage (before they get removed from tracking)
            for plate_id, plate_info in tracked_plates.items():
                best_text = plate_info.get('best_text', '')
                if best_text and validate_vietnamese_plate_format(best_text):
                    # Save or update this plate in confirmed storage
                    if plate_id not in confirmed_plates:
                        confirmed_plates[plate_id] = {
                            'plate_number': best_text,
                            'confidence': int(plate_info.get('best_confidence', 0)),
                            'plate_id': plate_info['id'],
                            'frame_count': plate_info['frame_count'],
                            'frequency': plate_info['text_history'].count(best_text) if plate_info.get('text_history') else 0,
                            'total_detections': len(plate_info.get('text_history', []))
                        }
                    else:
                        # Update with latest data (including plate_number in case it changed)
                        confirmed_plates[plate_id]['plate_number'] = best_text
                        confirmed_plates[plate_id]['confidence'] = int(plate_info.get('best_confidence', 0))
                        confirmed_plates[plate_id]['frame_count'] = plate_info['frame_count']
                        confirmed_plates[plate_id]['frequency'] = plate_info['text_history'].count(best_text) if plate_info.get('text_history') else 0
                        confirmed_plates[plate_id]['total_detections'] = len(plate_info.get('text_history', []))
            
            # Print progress every 100 frames
            if frames_processed % 100 == 0:
                progress = (frames_processed / total_frames) * 100
                logger.info(f"Progress: {frames_processed}/{total_frames} ({progress:.1f}%)")
        
        cap.release()
        out.release()
        
        # Use confirmed plates dictionary for final results
        all_detections = list(confirmed_plates.values())
        
        logger.info(f"=== FINAL TRACKING RESULTS ===")
        logger.info(f"Confirmed plates saved: {len(confirmed_plates)}")
        logger.info(f"Plates still in tracker: {len(tracked_plates)}")
        
        logger.info(f"\n📋 DETECTED LICENSE PLATES:")
        logger.info(f"{'='*60}")
        for i, (plate_id, detection) in enumerate(confirmed_plates.items(), 1):
            logger.info(f"{i}. {detection['plate_number']} ({detection['confidence']}% confidence)")
        logger.info(f"{'='*60}")
        
        logger.info(f"=== END TRACKING RESULTS ===")
        logger.info(f"Video processing complete: {len(all_detections)} unique plates detected in {frames_processed} frames")
        logger.info(f"Output video saved: {output_video_path}")
        
        # Verify output file exists and has content
        if not os.path.exists(output_video_path):
            logger.error(f"Output video file not created: {output_video_path}")
            raise RuntimeError("Failed to create output video file")
        
        file_size = os.path.getsize(output_video_path)
        logger.info(f"Output video size: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1000:  # Less than 1KB probably means error
            logger.error(f"Output video file too small: {file_size} bytes")
            raise RuntimeError("Output video file is corrupted or empty")
        
        # Clean up input temp file
        try:
            os.remove(temp_video_path)
        except Exception as e:
            logger.warning(f"Failed to remove temp video file: {e}")
        
        processing_time = round(time.time() - start_time, 2)
        
        # Log what we're sending to frontend
        logger.info(f"=== SENDING TO FRONTEND ===")
        logger.info(f"total_detections: {len(all_detections)}")
        logger.info(f"detections array: {all_detections}")
        logger.info(f"===========================")
        
        return jsonify({
            'success': True,
            'detections': all_detections,
            'total_detections': len(all_detections),
            'frames_processed': frames_processed,
            'processing_time': processing_time,
            'video_fps': video_fps,
            'video_width': video_width,
            'video_height': video_height,
            'output_video': f'/uploads/{output_video_name}'
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Clean up temp files if they exist
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass
        
        if 'output_video_path' in locals() and os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
            except:
                pass
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recent-detections', methods=['GET'])
def recent_detections():
    """Get recent plate detections"""
    try:
        camera_id = request.args.get('camera_id', type=int)
        minutes = request.args.get('minutes', default=5, type=int)
        
        detections = plate_deduplication.get_recent_plates(camera_id, minutes * 60)
        
        # Convert datetime objects to strings
        for detection in detections:
            if 'timestamp' in detection and isinstance(detection['timestamp'], datetime):
                detection['timestamp'] = detection['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections)
        })
    except (ValueError, TypeError) as e:
        logger.error("Error getting recent detections: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/tracking-stats', methods=['GET'])
def tracking_stats():
    """Get tracking statistics"""
    try:
        camera_id = request.args.get('camera_id', type=int)
        
        stats = plate_deduplication.get_stats(camera_id)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except (RuntimeError, ValueError) as e:
        logger.error("Error getting tracking stats: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear deduplication cache and session trackers"""
    try:
        data = request.json or {}
        camera_id = data.get('camera_id')
        
        plate_deduplication.clear_cache(camera_id)
        
        # Clear session trackers
        if camera_id:
            if camera_id in camera_trackers:
                camera_trackers[camera_id].reset()
        else:
            for tracker in camera_trackers.values():
                tracker.reset()
        
        return jsonify({
            'success': True,
            'message': f"Cache cleared for camera {camera_id}" if camera_id else "All caches cleared"
        })
    except (RuntimeError, ValueError) as e:
        logger.error("Error clearing cache: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/persist-detection', methods=['POST'])
def persist_detection():
    """Manually persist a plate detection"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['plate_number', 'camera_id', 'direction']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f"Missing required field: {field}"
                }), 400
        
        # Create dummy frame (in real scenario, you'd have the actual frame)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = plate_persistence.persist_plate_detection(
            frame=frame,
            camera_id=data['camera_id'],
            plate_number=data['plate_number'],
            confidence=data.get('confidence', 100),
            direction=data['direction'],
            tracking_id=data.get('tracking_id', 0),
            is_duplicate=data.get('is_duplicate', False)
        )
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except (ValueError, IOError) as e:
        logger.error("Error persisting detection: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """Run cleanup operations"""
    try:
        data = request.json or {}
        days = data.get('days', 7)
        
        results = {
            'local_files': 0,
            'drive_folders': 0,
            'sheets': 0
        }
        
        if data.get('cleanup_local', True):
            results['local_files'] = plate_persistence.cleanup_old_files(days)
        
        if data.get('cleanup_drive', True):
            results['drive_folders'] = plate_persistence.cleanup_old_drive_folders(days)
        
        if data.get('cleanup_sheets', True):
            results['sheets'] = plate_persistence.cleanup_old_sheets(days)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except (IOError, RuntimeError) as e:
        logger.error("Error during cleanup: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/captures/<path:filename>')
def serve_capture(filename):
    """Serve captured images"""
    captures_dir = os.path.join(os.path.dirname(__file__), 'captures')
    return send_from_directory(captures_dir, filename)


@app.errorhandler(404)
def not_found(_error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error("Internal server error: %s", str(error))
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


def main():
    """Main entry point"""
    global models_initialized
    
    # Run startup checks
    try:
        from utils.startup_checks import run_startup_checks
        if not run_startup_checks():
            logger.error("Startup checks failed. Please fix errors and restart.")
            sys.exit(1)
    except ImportError:
        logger.warning("Startup checks module not available, skipping validation")
    
    # Initialize database
    try:
        from init_database import init_database
        init_database()
    except (ImportError, Exception) as e:
        logger.warning("Database auto-initialization skipped: %s", str(e))
    
    # Initialize models on startup
    logger.info("Initializing plate detection models...")
    models_initialized = init_alpr_models()
    if models_initialized:
        logger.info("Models initialized successfully")
    else:
        logger.warning("Failed to initialize models on startup. Will retry on first request.")
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info("Starting Plate Detection Web Server on %s:%d", host, port)
    logger.info("Optimizations enabled: async persistence, thread-safe camera manager, centralized constants")
    
    try:
        # Run Flask app
        app.run(host=host, port=port, debug=debug, threaded=True)
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down server...")
        
        # Release all cameras using CameraManager
        camera_manager.release_all()
        logger.info("All cameras released")
        
        # Shutdown thread pools
        logger.info("Shutting down thread pools...")
        persistence_executor.shutdown(wait=True, cancel_futures=False)
        cleanup_executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Thread pools shut down")
        
        # Stop deduplication cleanup scheduler
        plate_deduplication.stop_cleanup_scheduler()
        logger.info("Cleanup scheduler stopped")


if __name__ == '__main__':
    main()
