# Plate Detector

License plate detection and recognition system project.

## Features

- **YOLO-based vehicle detection** with ALPR (Automatic License Plate Recognition)
- **Plate deduplication** to avoid duplicate detections
- **Google Drive integration** for image storage
- **Google Sheets integration** for plate logging
- **SQLite database** for local persistence
- **Configurable detection settings** per camera
- **Web Server Application** with dashboard for monitoring and management

## Web Server Application

A Flask-based web application is available for easy plate detection and monitoring!

### Quick Start

```cmd
start_server.bat
```

This will:
- ✅ Run comprehensive health checks
- ✅ Auto-install missing packages
- ✅ Initialize database automatically
- ✅ Start the server on http://localhost:5000

**Or manual start:**
```cmd
python app.py
```

For detailed instructions, see:
- **[QUICKSTART.md](QUICKSTART.md)** - Quick setup guide

### Web Server Features

- **📹 Live Camera Feed** - Real-time detection from USB/IP cameras
- 📊 Real-time statistics dashboard
- 🕐 Recent detections monitoring
- 🎯 Multi-camera support
- 🔍 RESTful API endpoints
- 🧹 Cache and cleanup management
- ✅ Auto-install missing packages
- 🏥 Health check diagnostics
- 🛡️ Comprehensive error handling
- 🚀 Automatic database setup

## Project Structure

```
plate_detector/
├── app.py                          # Flask web server
├── health_check.py                 # System diagnostics & auto-install
├── init_database.py                # Auto database initialization
├── start_server.bat                # Enhanced startup script
├── utils/
│   ├── plate_detection.py          # Main detection logic
│   ├── plate_persistence.py        # Database & Google integration
│   ├── plate_deduplication.py      # Deduplication logic
│   ├── camera_manager.py           # Camera management
│   ├── error_handlers.py           # Validation utilities
│   ├── startup_checks.py           # Pre-flight checks
│   ├── plate_detection_config.json # Configuration
│   ├── plate_detection_credentials.json # API credentials
│   └── models/                     # YOLO models
│       ├── best.pt                 # Custom trained model
│       └── yolo11n.pt              # Base YOLO model
├── tests/
│   ├── test_plate_model.py         # Model testing
│   └── test_plate_visualization.py # Visualization testing
├── docs/
│   └── PLATE_DETECTION_WORKFLOW.md # Workflow documentation
├── templates/                      # Web UI templates
│   ├── index.html
│   ├── camera.html
│   └── test.html
└── requirements.txt                # Python dependencies
```

## Installation

### Method 1: Automated (Recommended)

```bash
git clone https://github.com/OliverQueen168/plate_detector.git
cd plate_detector
start_server.bat
```

The startup script will:
- ✅ Check Python installation
- ✅ Run health checks
- ✅ **Auto-install all missing packages**
- ✅ Initialize database
- ✅ Start the web server

### Method 2: Manual Installation

### 1. Clone the repository

```bash
git clone https://github.com/OliverQueen168/plate_detector.git
cd plate_detector
```

### 2. Create virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

**All packages (including optional Google integration):**
```bash
pip install -r requirements.txt
```

**Note:** PyTorch with CUDA support will be installed automatically from the PyTorch repository.

### 4. Setup credentials (optional for Google integration)

Edit `utils/plate_detection_credentials.json` with your:
- Google Drive API credentials
- Google Sheets API credentials

### 5. Run health check

```bash
python health_check.py
```

This will verify all packages and system requirements.

## Configuration

Edit `utils/plate_detection_config.json`:

```json
{
    "cameras": {
        "1": {
            "enabled": true,
            "confidence_threshold": 0.5,
            "detection_region": null
        }
    },
    "deduplication": {
        "time_window_seconds": 10,
        "similarity_threshold": 0.85
    },
    "persistence": {
        "google_drive_enabled": true,
        "google_sheets_enabled": true,
        "local_db_enabled": true
    }
}
```

## Usage

### As a Python module

```python
from utils.plate_detection import PlateDetector

# Initialize detector
detector = PlateDetector(camera_id=1)

# Process frame
frame = cv2.imread('image.jpg')
results = detector.detect(frame)

for result in results:
    print(f"Plate: {result['plate_number']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Vehicle type: {result['vehicle_type']}")
```

### Run tests

```bash
# Test model
python tests/test_plate_model.py

# Test visualization
python tests/test_plate_visualization.py
```

## Dependencies

Main dependencies:

**Required:**
- **OpenCV** (4.7.0.72) - Computer vision
- **Ultralytics** (8.3.218) - YOLO models
- **PyTorch** (2.7.0+cu128) - Deep learning with CUDA support
- **torchvision** (0.22.0+cu128) - PyTorch vision library
- **fast-alpr** (>=0.3.0) - License plate recognition
- **onnxruntime** (>=1.16.0) - ML runtime
- **Flask** (>=2.3.2) - Web framework
- **numpy** (>=1.24.3,<2.0) - Numerical computing

**Optional (for Google integration):**
- **ImageHash** (>=4.3.0) - Advanced deduplication
- **PyDrive** (1.3.1) - Google Drive integration
- **gspread** (>=6.2.1) - Google Sheets integration
- **oauth2client** (>=4.1.3) - Google API authentication

See `requirements.txt` for complete list.

### Auto-Installation

The system will automatically install missing packages when you run:
```bash
python health_check.py
```

Or to skip auto-install:
```bash
python health_check.py --no-install
```

## API Credentials Setup

### Google Drive API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable Google Drive API
4. Create credentials (Service Account)
5. Download JSON key file
6. Update path in `plate_detection_credentials.json`

### Google Sheets API

1. Enable Google Sheets API in same project
2. Share your spreadsheet with service account email
3. Update spreadsheet ID in `plate_detection_credentials.json`

## Performance

- **Detection speed**: ~30-50ms per frame (GPU with CUDA)
- **Recognition accuracy**: ~95% (well-lit conditions)
- **Deduplication**: Prevents duplicate detections within configurable time window
- **Memory usage**: ~500MB-1GB (depends on model)
- **Startup time**: ~5-10 seconds (with health checks)
- **Auto-recovery**: Handles camera failures, missing files, and network issues

## Key Features

✅ **Comprehensive Error Handling** - Robust error detection and recovery  
✅ **Auto-Install Packages** - Automatically installs missing dependencies  
✅ **Health Check System** - Validates system before startup  
✅ **Database Auto-Setup** - Creates schema and indexes automatically  
✅ **Input Validation** - All user inputs validated and sanitized  
✅ **Enhanced Logging** - Better error messages and debugging  
✅ **Startup Checks** - Pre-flight validation of all components  
✅ **Error Recovery** - Auto-recovery from common failures  

## Troubleshooting

### Run health check
```bash
python health_check.py
```

This will:
- ✅ Check all package installations
- ✅ Verify system files exist
- ✅ Validate configuration
- ✅ Check database schema
- ✅ Detect GPU/CUDA availability
- ✅ Monitor disk space
- ✅ **Auto-install missing packages**

### Common Issues

#### Missing packages
**Solution:** Health check will auto-install them. Or manually:
```bash
pip install -r requirements.txt
```

#### Model not found
```bash
# Ensure model file exists at utils/models/best.pt
# Download if needed or train your own model
```

#### NumPy version conflict
```bash
pip install "numpy>=1.24.3,<2.0"
```

#### Ultralytics C3k2 error
```bash
pip install ultralytics==8.3.218
```

#### CUDA out of memory
- Reduce batch size in config
- Use smaller model (yolo11n vs yolo11x)
- Process at lower resolution

#### Google API errors
- Check credentials.json path
- Verify API is enabled in Google Cloud Console
- Check service account permissions

#### Port 5000 already in use
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Manual Database Initialization
```bash
python init_database.py
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Contact

- GitHub: [@OliverQueen168](https://github.com/counterloop168)
- Repository: [plate_detector](https://github.com/counterloop168/plate_detector)
