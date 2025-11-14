# 🚀 Quick Start Guide - Plate Detection Web Server

## Step 1: Automated Setup ⚡

The easiest way to get started:

```cmd
start_server.bat
```

**What it does:**
- ✅ Checks Python installation
- ✅ Runs comprehensive health checks
- ✅ Auto-installs missing packages (both required and optional)
- ✅ Initializes database automatically
- ✅ Creates necessary directories
- ✅ Starts the web server on http://localhost:5000

**That's it!** Everything is set up automatically.

---

## Step 2: Manual Setup (Alternative)

If you prefer manual installation:

### 1. Install Dependencies

```cmd
pip install -r requirements.txt
```

**Note:** PyTorch with CUDA support will be installed automatically.

### 2. Run Health Check (Optional)

```cmd
python health_check.py
```

This will:
- Verify all packages are installed
- Auto-install any missing packages
- Check system requirements
- Validate configuration

### 3. Start the Web Server

```cmd
python app.py
```

## Step 3: Access the Dashboard

Open your web browser and navigate to:

```
http://localhost:5000
```

**Available Pages:**
- **Home**: http://localhost:5000
- **Camera 1**: http://localhost:5000/camera/1
- **Camera 2**: http://localhost:5000/camera/2
- **Test API**: http://localhost:5000/test

## Step 4: Start Live Camera Detection

### Using the Web Interface:

1. **Select Camera Source** - Choose your camera (USB 0, 1, 2, or custom IP camera URL)
2. **Click "Start Camera"** - Begin real-time detection
3. **Monitor Detections** - Watch live feed with plate detections
4. **View Statistics** - See detection counts and recent plates

### Using the API Test Script:

```cmd
python test_api.py
```

## Step 5: Monitor Detections

The dashboard automatically shows:
- **System Status**: Models loaded, system online
- **Statistics**: Total detections, camera-specific counts
- **Recent Detections**: Last 5 minutes of detections

## Configuration

Edit `utils/plate_detection_config.json` to customize:
- Enabled cameras
- Detection FPS
- Confidence threshold
- Deduplication settings
- Storage settings

## Common Issues & Solutions

### Health Check Fails
```cmd
python health_check.py
```
The health check will auto-install missing packages.

### Missing Packages
**Automatic Fix:**
```cmd
python health_check.py
```

**Manual Fix:**
```cmd
pip install -r requirements.txt
```

### NumPy Version Error
```cmd
pip install "numpy>=1.24.3,<2.0"
```

### Ultralytics C3k2 Error
```cmd
pip install ultralytics==8.3.218
```

### Port Already in Use

**Windows:**
```cmd
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

Or change the port:
```cmd
set PORT=8080
python app.py
```

### Models Not Loading

Ensure the model file exists at:
```
utils/models/best.pt
```

### Database Not Found
```cmd
python init_database.py
```

### Import Errors

Add the project to Python path:

```cmd
set PYTHONPATH=%PYTHONPATH%;C:\New Volume\plate_detector
```

## API Endpoints

Once running, access these endpoints:

- **Dashboard**: http://localhost:5000
- **Camera 1**: http://localhost:5000/camera/1
- **Camera 2**: http://localhost:5000/camera/2
- **Start Camera**: POST http://localhost:5000/camera/start
- **Stop Camera**: POST http://localhost:5000/camera/stop
- **Camera Stream**: http://localhost:5000/api/camera/stream/1
- **Recent Detections**: http://localhost:5000/api/recent-detections
- **Process Video**: POST http://localhost:5000/api/process-video
- **Update Config**: POST http://localhost:5000/api/config

## Features

✅ **Auto-install packages** - Automatically installs missing dependencies  
✅ **Health check system** - Validates system before startup  
✅ **Live camera streaming** - Real-time video detection  
✅ **Real-time plate recognition** - Instant OCR processing  
✅ **Vietnamese plate format** - Optimized for Vietnamese plates  
✅ **Multi-camera support** - Camera 1 (IN) and Camera 2 (OUT)  
✅ **Detection statistics** - Track detections over time  
✅ **Recent detections** - Monitor latest plates  
✅ **Cache management** - Automatic cleanup  
✅ **Database auto-setup** - Creates schema automatically  
✅ **Error recovery** - Handles failures gracefully  
✅ **Google Drive/Sheets** - Optional cloud integration  

## New Tools

### Health Check
```cmd
python health_check.py
```
Comprehensive system diagnostics with auto-install.

### Database Initialization
```cmd
python init_database.py
```
Manually initialize database if needed.

### Enhanced Startup
```cmd
start_server.bat
```
One-click setup and launch with all validations.  

## Next Steps

- Review full documentation in `WEB_SERVER_README.md`
- Configure Google Drive/Sheets (see `utils/plate_detection_credentials.README.md`)
- Customize detection settings in config file
- Integrate with your camera system

## Need Help?

Check the detailed documentation:
- `README.md` - Project overview
- `docs/PLATE_DETECTION_WORKFLOW.md` - Detection workflow

---

**Enjoy using the Plate Detection Web Server! 🚗📸**
