"""
Configuration file for RTSP ALPR System
Customize these values instead of modifying the main code
"""

import os

# Model Configuration
MODEL_PATH = r"utils\models\best.pt"
OCR_MODEL = "cct-s-v1-global-model"  # OCR model name

# YOLO Detection Parameters
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections (0.0-1.0)
YOLO_IOU_THRESHOLD = 0.4  # IoU threshold for non-maximum suppression

# Stream Configuration
# RTSP_URL = "rtsp://192.168.0.126:8554/cam1"
# RECONNECT_DELAY_SECONDS = 5
# MAX_RECONNECT_ATTEMPTS = 999999

# Crop Parameters
CROP_X = 200  # Row position
CROP_Y = 830  # Column position
CROP_TIMES = 1  # Number of crop windows
CROP_SIZE = 640  # Size of each crop window (640x640)

# Output Configuration
OUTPUT_DIR = r"crop_plate"

# Display Configuration
MAX_DISPLAY_WIDTH = 1690
MAX_DISPLAY_HEIGHT = 923
SHOW_CROP_WINDOWS = True

# Logging Configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "alpr.log"
