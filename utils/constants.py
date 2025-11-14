"""
Constants for Plate Detection System
Centralized configuration values to avoid magic numbers
"""

# Frame Processing
FPS_TARGET = 30
FRAME_DELAY = 1.0 / FPS_TARGET  # ~0.033 seconds
MAX_FRAME_COPIES = 2

# Detection Regions
DETECTION_REGION_SIZE = 640  # pixels
CAMERA_1_REGION_X = 300
CAMERA_1_REGION_Y = 140
CAMERA_2_REGION_X = 200
CAMERA_2_REGION_Y = 140

# Tracking
TRACKING_DISTANCE_THRESHOLD = 50  # pixels
MOVEMENT_THRESHOLD_PX = 100  # pixels for significant movement
PLATE_TIMEOUT_FRAMES = 60  # frames before removing tracked plate
PLATE_TIMEOUT_SECONDS = 10  # seconds for time-based cleanup

# Deduplication
DEDUP_TIME_WINDOW_SECONDS = 5
DEDUP_HASH_THRESHOLD = 0.95
MIN_CONFIDENCE_THRESHOLD = 0.6
MIN_IMAGE_SAVE_INTERVAL_SECONDS = 300  # 5 minutes

# Text History
MAX_TEXT_HISTORY_SIZE = 10
MIN_VOTES_REQUIRED = 3

# Image Processing
JPEG_QUALITY = 85
JPEG_QUALITY_LOW = 70

# Cleanup
GC_INTERVAL_FRAMES = 100  # Run garbage collection every N frames

# Camera Configuration
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
CAMERA_FPS = 30

# Vietnamese Plate Format
MIN_PLATE_LENGTH = 8
MAX_PLATE_LENGTH = 10

# Performance
MAX_CACHED_HASHES = 1000
DB_CONNECTION_POOL_SIZE = 5
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1

# API
MAX_UPLOAD_SIZE_MB = 16
API_RATE_LIMIT_PER_MINUTE = 10
