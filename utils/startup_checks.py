"""
Startup Validation and Initialization
Performs comprehensive checks before starting the application
"""

import os
import sys
import logging
import sqlite3

logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher required, got %s", sys.version)
        return False
    logger.info("✅ Python version: %s", sys.version.split()[0])
    return True


def check_dependencies():
    """Check required dependencies are installed"""
    required_packages = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'flask': 'Flask',
        'numpy': 'numpy',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            logger.info("✅ %s installed", package_name)
        except ImportError:
            logger.error("❌ %s not installed", package_name)
            missing.append(package_name)
    
    if missing:
        logger.error("Missing packages: %s", ', '.join(missing))
        logger.error("Install with: pip install %s", ' '.join(missing))
        return False
    
    return True


def check_model_files():
    """Check model files exist"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'utils', 'models', 'best.pt')
    
    if not os.path.exists(model_path):
        logger.error("❌ Model file not found: %s", model_path)
        logger.error("Please ensure best.pt is in utils/models/ directory")
        return False
    
    file_size = os.path.getsize(model_path)
    logger.info("✅ Model file found: %.2f MB", file_size / (1024 * 1024))
    return True


def check_config_files():
    """Check configuration files exist"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, 'utils', 'plate_detection_config.json')
    
    if not os.path.exists(config_path):
        logger.warning("⚠️  Config file not found: %s", config_path)
        logger.warning("Will use default configuration")
        return True  # Non-critical
    
    logger.info("✅ Config file found")
    return True


def check_directories():
    """Check and create necessary directories"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    directories = [
        os.path.join(base_dir, 'captures'),
        os.path.join(base_dir, 'uploads'),
        os.path.join(base_dir, 'instance'),
        os.path.join(base_dir, 'templates'),
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info("✅ Directory ready: %s", os.path.basename(directory))
        except OSError as e:
            logger.error("❌ Failed to create directory %s: %s", directory, str(e))
            return False
    
    return True


def check_database():
    """Check database exists and has correct schema"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(base_dir, 'instance', 'site.db')
    
    if not os.path.exists(db_path):
        logger.warning("⚠️  Database not found, will be created on first use")
        return init_database(db_path)
    
    # Verify schema
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='plate_detection'
        """)
        
        if cursor.fetchone():
            logger.info("✅ Database schema verified")
            conn.close()
            return True
        else:
            logger.warning("⚠️  Table 'plate_detection' missing, will create")
            conn.close()
            return init_database(db_path)
            
    except sqlite3.Error as e:
        logger.error("❌ Database check failed: %s", str(e))
        return False


def init_database(db_path):
    """Initialize database with schema"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plate_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                camera_id INTEGER NOT NULL,
                direction TEXT NOT NULL,
                image_path TEXT,
                drive_link TEXT,
                drive_folder_link TEXT,
                is_duplicate INTEGER DEFAULT 0,
                tracking_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate_timestamp 
            ON plate_detection(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate_camera 
            ON plate_detection(camera_id, timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database initialized successfully")
        return True
        
    except sqlite3.Error as e:
        logger.error("❌ Database initialization failed: %s", str(e))
        return False


def check_port_available(port=5000):
    """Check if port is available"""
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            logger.info("✅ Port %d is available", port)
            return True
    except OSError:
        logger.error("❌ Port %d is already in use", port)
        logger.error("Please stop other services using this port or change PORT environment variable")
        return False


def check_cuda_available():
    """Check CUDA availability for GPU acceleration"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("✅ CUDA available: %s", device_name)
            return True
        else:
            logger.warning("⚠️  CUDA not available, will use CPU (slower)")
            return True  # Non-critical
    except ImportError:
        return True


def run_startup_checks():
    """Run all startup checks"""
    logger.info("=" * 60)
    logger.info("Starting Plate Detection System - Startup Checks")
    logger.info("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("Config Files", check_config_files),
        ("Directories", check_directories),
        ("Database", check_database),
        ("Port Availability", check_port_available),
        ("CUDA/GPU", check_cuda_available),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        logger.info("")
        logger.info("Checking: %s", check_name)
        try:
            if not check_func():
                all_passed = False
                logger.error("❌ %s check failed", check_name)
        except Exception as e:
            logger.error("❌ %s check error: %s", check_name, str(e))
            all_passed = False
    
    logger.info("")
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("✅ All startup checks passed!")
        logger.info("=" * 60)
        return True
    else:
        logger.error("❌ Some startup checks failed")
        logger.error("Please fix the errors above before starting the application")
        logger.info("=" * 60)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if run_startup_checks():
        print("\n✅ System is ready to start!")
        print("Run: python app.py")
    else:
        print("\n❌ System not ready, please fix errors above")
        sys.exit(1)
