"""
Plate Detection Deduplication Module
Handles duplicate detection filtering using time-window, hash-based, and confidence voting
Optimized with scheduled cleanup and hash caching
"""

import os
import time
import cv2
import json
import imagehash
from PIL import Image
import numpy as np
import logging
from threading import Lock, Thread
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Deduplication state
recent_plates = defaultdict(dict)  # {camera_id: {plate_number: {...}}}
last_saved_images = defaultdict(dict)  # {camera_id: {tracking_id: {...}}} - Track when images were last saved
dedup_lock = Lock()

# Cleanup scheduler
cleanup_running = False
cleanup_thread = None

# Load configuration
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'plate_detection_config.json')
config = {}

try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    config = {}

# Configuration with defaults
DEDUP_TIME_WINDOW = config.get('deduplication', {}).get('time_window_seconds', 5)
DEDUP_HASH_THRESHOLD = config.get('deduplication', {}).get('hash_threshold', 0.95)
MIN_CONFIDENCE_THRESHOLD = config.get('confidence_threshold', 0.6)
MIN_IMAGE_SAVE_INTERVAL = config.get('deduplication', {}).get('min_image_save_interval', 300)  # 5 minutes default

logger.info(f"Deduplication config: time_window={DEDUP_TIME_WINDOW}s, "
           f"hash_threshold={DEDUP_HASH_THRESHOLD}, min_confidence={MIN_CONFIDENCE_THRESHOLD}, "
           f"min_image_save_interval={MIN_IMAGE_SAVE_INTERVAL}s")


def scheduled_cleanup():
    """Background thread for periodic cache cleanup"""
    global cleanup_running
    
    while cleanup_running:
        try:
            time.sleep(30)  # Run every 30 seconds
            
            current_time = time.time()
            
            with dedup_lock:
                # Clean up recent_plates
                for camera_key in list(recent_plates.keys()):
                    plates_to_remove = []
                    for plate_num, plate_data in list(recent_plates[camera_key].items()):
                        if current_time - plate_data['timestamp'] > DEDUP_TIME_WINDOW * 2:
                            plates_to_remove.append(plate_num)
                    
                    for plate_num in plates_to_remove:
                        del recent_plates[camera_key][plate_num]
                
                # Clean up last_saved_images
                for camera_key in list(last_saved_images.keys()):
                    entries_to_remove = []
                    for tid, data in list(last_saved_images[camera_key].items()):
                        if current_time - data['timestamp'] > MIN_IMAGE_SAVE_INTERVAL * 2:
                            entries_to_remove.append(tid)
                    
                    for tid in entries_to_remove:
                        del last_saved_images[camera_key][tid]
                        
        except Exception as e:
            logger.error(f"Error in scheduled cleanup: {e}")


def start_cleanup_scheduler():
    """Start the background cleanup scheduler"""
    global cleanup_running, cleanup_thread
    
    if not cleanup_running:
        cleanup_running = True
        cleanup_thread = Thread(target=scheduled_cleanup, daemon=True, name="dedup-cleanup")
        cleanup_thread.start()
        logger.info("Deduplication cleanup scheduler started")


def stop_cleanup_scheduler():
    """Stop the background cleanup scheduler"""
    global cleanup_running, cleanup_thread
    
    if cleanup_running:
        cleanup_running = False
        if cleanup_thread:
            cleanup_thread.join(timeout=5)
        logger.info("Deduplication cleanup scheduler stopped")


def compute_image_hash(image):
    """Compute perceptual hash of an image"""
    try:
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
        
        # Compute perceptual hash
        phash = imagehash.phash(image)
        return phash
    except Exception as e:
        logger.error(f"Error computing image hash: {e}")
        return None


def hash_similarity(hash1, hash2):
    """Calculate similarity between two hashes (0-1 scale)"""
    if hash1 is None or hash2 is None:
        return 0.0
    
    # Calculate Hamming distance
    hamming_distance = hash1 - hash2
    
    # Convert to similarity (0-1, where 1 is identical)
    # perceptual hash is 64 bits by default
    max_distance = 64
    similarity = 1.0 - (hamming_distance / max_distance)
    
    return similarity


def is_duplicate(camera_id, plate_number, image_crop, confidence, bbox):
    """
    Check if a plate detection is a duplicate
    
    Args:
        camera_id: Camera identifier
        plate_number: Detected plate text
        image_crop: Cropped image of the plate (numpy array)
        confidence: Detection confidence (0-100)
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        (is_duplicate, reason) - tuple of boolean and reason string
    """
    
    # MODIFIED: Accept very low confidence for poor quality cameras
    # Only reject if confidence is 0 (no detection)
    if confidence <= 0:
        return True, f"No confidence: {confidence}%"
    
    with dedup_lock:
        camera_key = str(camera_id)
        
        # Initialize camera if not exists
        if camera_key not in recent_plates:
            recent_plates[camera_key] = {}
        
        current_time = time.time()
        
        # Note: Cleanup is now handled by scheduled_cleanup() - no per-call cleanup
        
        # Check if this plate was recently seen
        if plate_number in recent_plates[camera_key]:
            recent_data = recent_plates[camera_key][plate_number]
            time_diff = current_time - recent_data['timestamp']
            
            # Time-based deduplication
            if time_diff < DEDUP_TIME_WINDOW:
                # Check hash similarity if we have an image
                if image_crop is not None and recent_data.get('image_hash') is not None:
                    current_hash = compute_image_hash(image_crop)
                    if current_hash is not None:
                        similarity = hash_similarity(current_hash, recent_data['image_hash'])
                        
                        if similarity > DEDUP_HASH_THRESHOLD:
                            return True, f"Duplicate within {time_diff:.1f}s, hash similarity: {similarity:.2f}"
                
                # Even without hash check, if it's within time window, likely duplicate
                return True, f"Same plate within {time_diff:.1f}s"
        
        # Not a duplicate - store this detection
        image_hash = compute_image_hash(image_crop) if image_crop is not None else None
        
        recent_plates[camera_key][plate_number] = {
            'timestamp': current_time,
            'confidence': confidence,
            'bbox': bbox,
            'image_hash': image_hash,
            'datetime': datetime.now()
        }
        
        return False, "New detection"


def should_save_to_database(camera_id, plate_number, confidence, detection_count):
    """
    Determine if a plate detection should be saved to database
    MODIFIED: Save every valid detection immediately (no voting system)
    
    Args:
        camera_id: Camera identifier
        plate_number: Detected plate text
        confidence: Current confidence percentage (0-100)
        detection_count: Number of times this plate has been detected
    
    Returns:
        bool - True if should save to database
    """
    
    # For poor quality cameras, accept any detection that passed validation
    # The validation happens earlier (Vietnamese plate format check)
    # Just ensure it has some minimum confidence (>0)
    if confidence > 0:
        return True
    
    return False


def get_recent_plates(camera_id=None, minutes=5):
    """
    Get recent plate detections
    
    Args:
        camera_id: Optional camera filter
        minutes: Look back this many minutes
    
    Returns:
        List of recent plate detections
    """
    with dedup_lock:
        cutoff_time = time.time() - (minutes * 60)
        results = []
        
        cameras_to_check = [str(camera_id)] if camera_id else recent_plates.keys()
        
        for cam_key in cameras_to_check:
            if cam_key in recent_plates:
                for plate_num, plate_data in recent_plates[cam_key].items():
                    if plate_data['timestamp'] > cutoff_time:
                        results.append({
                            'camera_id': int(cam_key),
                            'plate_number': plate_num,
                            'confidence': plate_data['confidence'],
                            'timestamp': plate_data['datetime'],
                            'bbox': plate_data['bbox']
                        })
        
        # Sort by timestamp descending
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        return results


def clear_cache(camera_id=None):
    """Clear deduplication cache"""
    with dedup_lock:
        if camera_id:
            camera_key = str(camera_id)
            if camera_key in recent_plates:
                recent_plates[camera_key] = {}
        else:
            recent_plates.clear()


def get_stats(camera_id=None):
    """Get deduplication statistics"""
    with dedup_lock:
        if camera_id:
            camera_key = str(camera_id)
            if camera_key in recent_plates:
                return {
                    'camera_id': camera_id,
                    'cached_plates': len(recent_plates[camera_key]),
                    'plates': list(recent_plates[camera_key].keys())
                }
            return {'camera_id': camera_id, 'cached_plates': 0, 'plates': []}
        
        # All cameras
        total = sum(len(plates) for plates in recent_plates.values())
        return {
            'total_cached_plates': total,
            'cameras': {
                int(cam_key): {
                    'cached_plates': len(plates),
                    'plates': list(plates.keys())
                }
                for cam_key, plates in recent_plates.items()
            }
        }


def should_save_image(camera_id, tracking_id, plate_number, bbox=None):
    """
    Determine if we should save an image for this detection
    Prevents saving multiple images of the same stationary vehicle
    
    Args:
        camera_id: Camera identifier
        tracking_id: Tracking ID of the plate
        plate_number: Detected plate text
        bbox: Bounding box [x1, y1, x2, y2] (optional, for position tracking)
    
    Returns:
        (should_save, reason) - tuple of boolean and reason string
    """
    with dedup_lock:
        camera_key = str(camera_id)
        
        # Initialize camera if not exists
        if camera_key not in last_saved_images:
            last_saved_images[camera_key] = {}
        
        current_time = time.time()
        
        # Note: Cleanup is now handled by scheduled_cleanup() - no per-call cleanup
        
        # Check if this tracking ID was recently saved
        if tracking_id in last_saved_images[camera_key]:
            last_save_data = last_saved_images[camera_key][tracking_id]
            time_since_last_save = current_time - last_save_data['timestamp']
            
            # If we saved recently, check if plate has moved significantly
            if time_since_last_save < MIN_IMAGE_SAVE_INTERVAL:
                # If we have bbox data, check if position changed significantly
                if bbox and last_save_data.get('bbox'):
                    # Calculate center point movement
                    last_bbox = last_save_data['bbox']
                    last_center_x = (last_bbox[0] + last_bbox[2]) / 2
                    last_center_y = (last_bbox[1] + last_bbox[3]) / 2
                    curr_center_x = (bbox[0] + bbox[2]) / 2
                    curr_center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Calculate distance moved
                    distance_moved = ((curr_center_x - last_center_x)**2 + 
                                     (curr_center_y - last_center_y)**2)**0.5
                    
                    # If moved more than 100 pixels, consider it a new position
                    if distance_moved > 100:
                        logger.info(f"[Cam {camera_id}] Plate {tracking_id} moved {distance_moved:.1f}px, saving new image")
                        # Update and allow save
                        last_saved_images[camera_key][tracking_id] = {
                            'timestamp': current_time,
                            'plate_number': plate_number,
                            'bbox': bbox,
                            'datetime': datetime.now()
                        }
                        return True, f"Plate moved {distance_moved:.1f}px"
                
                # Still in same position, don't save
                return False, f"Image saved {time_since_last_save:.1f}s ago (waiting {MIN_IMAGE_SAVE_INTERVAL}s)"
        
        # First time seeing this tracking ID or enough time has passed - save it
        last_saved_images[camera_key][tracking_id] = {
            'timestamp': current_time,
            'plate_number': plate_number,
            'bbox': bbox,
            'datetime': datetime.now()
        }
        
        return True, "New tracking ID or sufficient time elapsed"


def get_recent_plates(camera_id=None, time_window=60):
    """
    Get recent plate detections within time window
    
    Args:
        camera_id: Optional camera filter
        time_window: Time window in seconds (default 60)
    
    Returns:
        list: List of recent plate detections
    """
    with dedup_lock:
        results = []
        cutoff_time = time.time() - time_window
        
        cameras_to_check = [str(camera_id)] if camera_id else recent_plates.keys()
        
        for cam_key in cameras_to_check:
            if cam_key in recent_plates:
                for plate_num, plate_data in recent_plates[cam_key].items():
                    if plate_data['timestamp'] > cutoff_time:
                        results.append({
                            'camera_id': int(cam_key),
                            'plate_number': plate_num,
                            'confidence': plate_data['confidence'],
                            'timestamp': plate_data['datetime'],
                            'bbox': plate_data['bbox']
                        })
        
        # Sort by timestamp descending
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        return results


# Start cleanup scheduler on module load
start_cleanup_scheduler()

logger.info("Plate deduplication module loaded")
