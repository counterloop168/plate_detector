"""
Comprehensive Error Handlers and Validators
Provides robust error handling and input validation
"""

import os
import logging
from functools import wraps
from typing import Any, Callable, Optional
import json

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    pass


class ConfigurationError(Exception):
    """Custom configuration error"""
    pass


class ModelError(Exception):
    """Custom model error"""
    pass


def safe_execute(default_return=None, exceptions=(Exception,)):
    """
    Decorator for safe function execution with error handling
    
    Args:
        default_return: Value to return on error
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.error("Error in %s: %s", func.__name__, str(e))
                return default_return
        return wrapper
    return decorator


def validate_camera_id(camera_id: Any) -> int:
    """
    Validate camera ID
    
    Args:
        camera_id: Camera identifier to validate
    
    Returns:
        int: Validated camera ID
    
    Raises:
        ValidationError: If camera ID is invalid
    """
    try:
        cam_id = int(camera_id)
        if cam_id not in [1, 2]:
            raise ValidationError(f"Camera ID must be 1 or 2, got {cam_id}")
        return cam_id
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid camera ID format: {str(e)}") from e


def validate_confidence(confidence: Any) -> float:
    """
    Validate confidence score
    
    Args:
        confidence: Confidence value to validate
    
    Returns:
        float: Validated confidence (0.0 to 1.0)
    
    Raises:
        ValidationError: If confidence is invalid
    """
    try:
        conf = float(confidence)
        if not 0.0 <= conf <= 100.0:
            raise ValidationError(f"Confidence must be between 0 and 100, got {conf}")
        return conf
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid confidence format: {str(e)}") from e


def validate_plate_number(plate_number: str) -> str:
    """
    Validate plate number format
    
    Args:
        plate_number: Plate number to validate
    
    Returns:
        str: Validated plate number
    
    Raises:
        ValidationError: If plate number is invalid
    """
    if not plate_number or not isinstance(plate_number, str):
        raise ValidationError("Plate number must be a non-empty string")
    
    # Clean plate number
    clean_plate = plate_number.strip()
    
    if len(clean_plate) < 6:
        raise ValidationError(f"Plate number too short: {clean_plate}")
    
    if len(clean_plate) > 15:
        raise ValidationError(f"Plate number too long: {clean_plate}")
    
    return clean_plate


def validate_direction(direction: str) -> str:
    """
    Validate direction value
    
    Args:
        direction: Direction to validate
    
    Returns:
        str: Validated direction ('IN' or 'OUT')
    
    Raises:
        ValidationError: If direction is invalid
    """
    if not direction or not isinstance(direction, str):
        raise ValidationError("Direction must be a non-empty string")
    
    direction_upper = direction.strip().upper()
    
    if direction_upper not in ['IN', 'OUT']:
        raise ValidationError(f"Direction must be 'IN' or 'OUT', got {direction}")
    
    return direction_upper


def validate_bbox(bbox: Any) -> list:
    """
    Validate bounding box
    
    Args:
        bbox: Bounding box to validate [x1, y1, x2, y2]
    
    Returns:
        list: Validated bounding box
    
    Raises:
        ValidationError: If bounding box is invalid
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValidationError("Bounding box must be a list/tuple of 4 coordinates")
    
    try:
        x1, y1, x2, y2 = [int(x) for x in bbox]
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid bounding box coordinates: {str(e)}") from e
    
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise ValidationError(f"Bounding box coordinates must be non-negative: {bbox}")
    
    if x1 >= x2 or y1 >= y2:
        raise ValidationError(f"Invalid bounding box dimensions: {bbox}")
    
    return [x1, y1, x2, y2]


def validate_config(config: dict) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration to validate
    
    Returns:
        bool: True if valid
    
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a dictionary")
    
    # Validate enabled_cameras
    if 'enabled_cameras' in config:
        if not isinstance(config['enabled_cameras'], list):
            raise ConfigurationError("enabled_cameras must be a list")
        
        for cam_id in config['enabled_cameras']:
            try:
                validate_camera_id(cam_id)
            except ValidationError as e:
                raise ConfigurationError(f"Invalid camera ID in config: {str(e)}") from e
    
    # Validate detection_fps
    if 'detection_fps' in config:
        fps = config['detection_fps']
        try:
            fps_val = float(fps)
            if fps_val <= 0 or fps_val > 60:
                raise ConfigurationError(f"detection_fps must be between 0 and 60, got {fps}")
        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Invalid detection_fps format: {str(e)}") from e
    
    # Validate confidence_threshold
    if 'confidence_threshold' in config:
        try:
            conf = float(config['confidence_threshold'])
            if not 0.0 <= conf <= 1.0:
                raise ConfigurationError(f"confidence_threshold must be between 0 and 1, got {conf}")
        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Invalid confidence_threshold format: {str(e)}") from e
    
    logger.info("✅ Configuration validated successfully")
    return True


def validate_model_file(model_path: str) -> bool:
    """
    Validate model file exists and is readable
    
    Args:
        model_path: Path to model file
    
    Returns:
        bool: True if valid
    
    Raises:
        ModelError: If model file is invalid
    """
    if not model_path:
        raise ModelError("Model path is empty")
    
    if not os.path.exists(model_path):
        raise ModelError(f"Model file not found: {model_path}")
    
    if not os.path.isfile(model_path):
        raise ModelError(f"Model path is not a file: {model_path}")
    
    if not os.access(model_path, os.R_OK):
        raise ModelError(f"Model file is not readable: {model_path}")
    
    # Check file size (should be > 1MB for valid model)
    file_size = os.path.getsize(model_path)
    if file_size < 1024 * 1024:  # 1MB
        raise ModelError(f"Model file too small ({file_size} bytes), may be corrupted: {model_path}")
    
    logger.info("✅ Model file validated: %s (%.2f MB)", model_path, file_size / (1024 * 1024))
    return True


def load_json_safe(file_path: str, default: Optional[dict] = None) -> dict:
    """
    Safely load JSON file with error handling
    
    Args:
        file_path: Path to JSON file
        default: Default value to return on error
    
    Returns:
        dict: Loaded JSON data or default
    """
    try:
        if not os.path.exists(file_path):
            logger.warning("JSON file not found: %s, using default", file_path)
            return default or {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info("✅ Loaded JSON: %s", file_path)
            return data
            
    except json.JSONDecodeError as e:
        logger.error("JSON decode error in %s: %s", file_path, str(e))
        return default or {}
    except IOError as e:
        logger.error("IO error reading %s: %s", file_path, str(e))
        return default or {}


def save_json_safe(file_path: str, data: dict) -> bool:
    """
    Safely save JSON file with error handling
    
    Args:
        file_path: Path to JSON file
        data: Data to save
    
    Returns:
        bool: True if successful
    """
    try:
        # Create directory if not exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            logger.info("✅ Saved JSON: %s", file_path)
            return True
            
    except (IOError, TypeError) as e:
        logger.error("Error saving JSON to %s: %s", file_path, str(e))
        return False


def check_disk_space(path: str, required_mb: int = 100) -> bool:
    """
    Check if sufficient disk space is available
    
    Args:
        path: Path to check
        required_mb: Required space in MB
    
    Returns:
        bool: True if sufficient space
    """
    try:
        import shutil
        stat = shutil.disk_usage(path)
        available_mb = stat.free / (1024 * 1024)
        
        if available_mb < required_mb:
            logger.warning("⚠️  Low disk space: %.2f MB available, %d MB required", 
                         available_mb, required_mb)
            return False
        
        logger.info("✅ Sufficient disk space: %.2f MB available", available_mb)
        return True
        
    except Exception as e:
        logger.error("Error checking disk space: %s", str(e))
        return True  # Assume OK if check fails


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks
    
    Args:
        filename: Filename to sanitize
    
    Returns:
        str: Sanitized filename
    """
    # Remove directory separators
    clean = filename.replace('/', '_').replace('\\', '_')
    # Remove dangerous characters
    clean = ''.join(c for c in clean if c.isalnum() or c in '.-_ ')
    # Limit length
    return clean[:255]


logger.info("Error handlers module loaded")
