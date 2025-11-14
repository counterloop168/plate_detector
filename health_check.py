"""
System Health Check Script
Comprehensive diagnostics for the plate detection system
"""

import os
import sys
import json
import logging
import sqlite3
import subprocess
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def install_package(package_name):
    """Install a Python package using pip"""
    try:
        logger.info("⏳ Installing %s...", package_name)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            logger.info("✅ Successfully installed %s", package_name)
            return True
        else:
            logger.error("❌ Failed to install %s: %s", package_name, result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Installation timeout for %s", package_name)
        return False
    except Exception as e:
        logger.error("❌ Error installing %s: %s", package_name, str(e))
        return False


def check_file_exists(path, name):
    """Check if file exists and report status"""
    if os.path.exists(path):
        if os.path.isfile(path):
            size = os.path.getsize(path)
            logger.info("✅ %s: Found (%.2f KB)", name, size / 1024)
            return True
        else:
            logger.error("❌ %s: Path exists but is not a file", name)
            return False
    else:
        logger.error("❌ %s: Not found at %s", name, path)
        return False


def check_directory(path, name):
    """Check if directory exists"""
    if os.path.exists(path) and os.path.isdir(path):
        file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        logger.info("✅ %s: Found (%d files)", name, file_count)
        return True
    else:
        logger.warning("⚠️  %s: Not found (will be created)", name)
        return False


def check_python_packages(auto_install=None):
    """Check required Python packages and optionally install missing ones"""
    # Use global flag if not explicitly provided
    if auto_install is None:
        import builtins
        auto_install = getattr(builtins, 'AUTO_INSTALL_PACKAGES', True)
    
    logger.info("\n" + "=" * 60)
    logger.info("PYTHON PACKAGES")
    if auto_install:
        logger.info("(Auto-install enabled)")
    logger.info("=" * 60)
    
    # Required packages - must be installed
    required_packages = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'ultralytics': 'ultralytics',
        'numpy': 'numpy',
        'flask': 'Flask',
        'PIL': 'Pillow',
        'fast_alpr': 'fast-alpr',
        'onnxruntime': 'onnxruntime',
    }
    
    # Optional packages - for Google Drive/Sheets integration
    optional_packages = {
        'imagehash': 'ImageHash',
        'pydrive': 'PyDrive',
        'gspread': 'gspread',
        'oauth2client': 'oauth2client',
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for module_name, package_name in required_packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            logger.info("✅ %s: %s", package_name, version)
        except ImportError:
            logger.warning("⚠️  %s: Not installed (REQUIRED)", package_name)
            missing_required.append(package_name)
    
    # Check optional packages
    for module_name, package_name in optional_packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            logger.info("✅ %s: %s", package_name, version)
        except ImportError:
            logger.warning("⚠️  %s: Not installed (optional - for Google integration)", package_name)
            missing_optional.append(package_name)
    
    # Auto-install missing packages if enabled
    if auto_install and (missing_required or missing_optional):
        logger.info("\n" + "=" * 60)
        logger.info("AUTO-INSTALLING MISSING PACKAGES")
        logger.info("=" * 60)
        
        # Install required packages
        if missing_required:
            logger.info("\n📦 Installing REQUIRED packages...")
            for package_name in missing_required:
                if install_package(package_name):
                    logger.info("✅ %s installed successfully", package_name)
                else:
                    logger.error("❌ Failed to install %s", package_name)
        
        # Install optional packages
        if missing_optional:
            logger.info("\n📦 Installing OPTIONAL packages...")
            for package_name in missing_optional:
                if install_package(package_name):
                    logger.info("✅ %s installed successfully", package_name)
                else:
                    logger.warning("⚠️  Failed to install optional package %s", package_name)
        
        # Re-check after installation
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION AFTER INSTALLATION")
        logger.info("=" * 60)
        
        all_packages = {**required_packages, **optional_packages}
        all_installed = True
        
        for module_name, package_name in all_packages.items():
            try:
                # Force reload to pick up newly installed packages
                if module_name in sys.modules:
                    del sys.modules[module_name]
                module = __import__(module_name)
                version = getattr(module, '__version__', 'unknown')
                logger.info("✅ %s: %s", package_name, version)
            except ImportError:
                if package_name in required_packages.values():
                    logger.error("❌ %s: Still not installed (REQUIRED)", package_name)
                    all_installed = False
                else:
                    logger.warning("⚠️  %s: Still not installed (optional)", package_name)
        
        return all_installed
    
    # If not auto-installing, just check if all required are present
    return len(missing_required) == 0


def check_system_files():
    """Check system files and directories"""
    logger.info("\n" + "=" * 60)
    logger.info("SYSTEM FILES & DIRECTORIES")
    logger.info("=" * 60)
    
    base_dir = os.path.dirname(__file__)
    
    files_to_check = {
        'Main App': os.path.join(base_dir, 'app.py'),
        'Requirements': os.path.join(base_dir, 'requirements.txt'),
        'Plate Detection': os.path.join(base_dir, 'utils', 'plate_detection.py'),
        'Persistence': os.path.join(base_dir, 'utils', 'plate_persistence.py'),
        'Deduplication': os.path.join(base_dir, 'utils', 'plate_deduplication.py'),
        'Camera Manager': os.path.join(base_dir, 'utils', 'camera_manager.py'),
        'Config': os.path.join(base_dir, 'utils', 'plate_detection_config.json'),
        'Model File': os.path.join(base_dir, 'utils', 'models', 'best.pt'),
    }
    
    all_found = True
    for name, path in files_to_check.items():
        if not check_file_exists(path, name):
            all_found = False
    
    # Check directories
    logger.info("")
    dirs_to_check = {
        'Captures': os.path.join(base_dir, 'captures'),
        'Uploads': os.path.join(base_dir, 'uploads'),
        'Instance': os.path.join(base_dir, 'instance'),
        'Templates': os.path.join(base_dir, 'templates'),
        'Utils': os.path.join(base_dir, 'utils'),
    }
    
    for name, path in dirs_to_check.items():
        check_directory(path, name)
    
    return all_found


def check_configuration():
    """Check configuration file"""
    logger.info("\n" + "=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    
    config_path = os.path.join(os.path.dirname(__file__), 'utils', 'plate_detection_config.json')
    
    if not os.path.exists(config_path):
        logger.error("❌ Config file not found")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info("✅ Config file loaded successfully")
        logger.info("")
        
        # Check key settings
        logger.info("Settings:")
        logger.info("  Enabled Cameras: %s", config.get('enabled_cameras', []))
        logger.info("  Detection FPS: %s", config.get('detection_fps', 'N/A'))
        logger.info("  Confidence Threshold: %s", config.get('confidence_threshold', 'N/A'))
        logger.info("  Google Drive Enabled: %s", config.get('google_drive', {}).get('enabled', False))
        logger.info("  Google Sheets Enabled: %s", config.get('google_sheets', {}).get('enabled', False))
        
        return True
        
    except json.JSONDecodeError as e:
        logger.error("❌ Config file has JSON errors: %s", str(e))
        return False
    except Exception as e:
        logger.error("❌ Error reading config: %s", str(e))
        return False


def check_database():
    """Check database"""
    logger.info("\n" + "=" * 60)
    logger.info("DATABASE")
    logger.info("=" * 60)
    
    db_path = os.path.join(os.path.dirname(__file__), 'instance', 'site.db')
    
    if not os.path.exists(db_path):
        logger.warning("⚠️  Database not found (will be created on first use)")
        return True
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        logger.info("✅ Database found")
        logger.info("Tables: %s", [t[0] for t in tables])
        
        # Check plate_detection table
        if ('plate_detection',) in tables:
            cursor.execute("SELECT COUNT(*) FROM plate_detection")
            count = cursor.fetchone()[0]
            logger.info("  plate_detection: %d records", count)
            
            # Get recent detections
            cursor.execute("""
                SELECT plate_number, camera_id, timestamp 
                FROM plate_detection 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent = cursor.fetchall()
            
            if recent:
                logger.info("")
                logger.info("Recent Detections:")
                for plate, cam, ts in recent:
                    logger.info("  - %s (Camera %d) at %s", plate, cam, ts)
        else:
            logger.warning("⚠️  Table 'plate_detection' not found")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        logger.error("❌ Database error: %s", str(e))
        return False


def check_gpu():
    """Check GPU availability"""
    logger.info("\n" + "=" * 60)
    logger.info("GPU / CUDA")
    logger.info("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info("✅ CUDA is available")
            logger.info("GPU Count: %d", device_count)
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                logger.info("")
                logger.info("GPU %d:", i)
                logger.info("  Name: %s", props.name)
                logger.info("  Memory: %.2f GB", props.total_memory / (1024**3))
                logger.info("  Compute Capability: %d.%d", props.major, props.minor)
            
            return True
        else:
            logger.warning("⚠️  CUDA not available")
            logger.warning("Detection will run on CPU (slower)")
            return True
            
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False


def check_disk_space():
    """Check disk space"""
    logger.info("\n" + "=" * 60)
    logger.info("DISK SPACE")
    logger.info("=" * 60)
    
    try:
        import shutil
        
        base_dir = os.path.dirname(__file__)
        stat = shutil.disk_usage(base_dir)
        
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        free_gb = stat.free / (1024**3)
        used_pct = (stat.used / stat.total) * 100
        
        logger.info("Total: %.2f GB", total_gb)
        logger.info("Used: %.2f GB (%.1f%%)", used_gb, used_pct)
        logger.info("Free: %.2f GB", free_gb)
        
        if free_gb < 1:
            logger.error("❌ Very low disk space (< 1 GB free)")
            return False
        elif free_gb < 5:
            logger.warning("⚠️  Low disk space (< 5 GB free)")
        else:
            logger.info("✅ Sufficient disk space")
        
        return True
        
    except Exception as e:
        logger.error("❌ Error checking disk space: %s", str(e))
        return False


def run_health_check():
    """Run complete health check"""
    print("\n" + "=" * 60)
    print("PLATE DETECTION SYSTEM - HEALTH CHECK")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    checks = [
        check_python_packages,
        check_system_files,
        check_configuration,
        check_database,
        check_gpu,
        check_disk_space,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            logger.error("❌ Check failed with error: %s", str(e))
            results.append(False)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    logger.info("Checks Passed: %d / %d", passed, total)
    
    if all(results):
        logger.info("✅ All checks passed - System is healthy!")
        return True
    else:
        logger.warning("⚠️  Some checks failed or have warnings")
        logger.warning("Review the messages above for details")
        return False


if __name__ == "__main__":
    try:
        # Check for command-line argument
        auto_install = True
        if len(sys.argv) > 1:
            if sys.argv[1] == "--no-install":
                auto_install = False
                logger.info("Auto-install disabled by user")
            elif sys.argv[1] == "--help":
                print("\nUsage: python health_check.py [OPTIONS]")
                print("\nOptions:")
                print("  --no-install    Skip automatic package installation")
                print("  --help          Show this help message")
                print("\nDefault: Auto-install missing packages\n")
                sys.exit(0)
        
        # Set auto_install flag globally for check_python_packages
        import builtins
        builtins.AUTO_INSTALL_PACKAGES = auto_install
        
        healthy = run_health_check()
        print("\n" + "=" * 60)
        if healthy:
            print("✅ SYSTEM READY")
            print("You can start the application with: python app.py")
        else:
            print("⚠️  SYSTEM HAS ISSUES")
            print("Please address the warnings/errors above")
        print("=" * 60 + "\n")
        
        sys.exit(0 if healthy else 1)
        
    except KeyboardInterrupt:
        print("\n\nHealth check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Health check failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
