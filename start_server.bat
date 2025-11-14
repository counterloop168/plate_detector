@echo off
setlocal enabledelayedexpansion
REM Enhanced Plate Detection Web Server Launcher with Health Checks
REM Automatically checks system health before starting

echo ========================================
echo  Plate Detection Web Server
echo  Enhanced Startup with Health Checks
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Run health check
echo Running system health check...
echo (Will auto-install missing packages)
echo.
python health_check.py
set HEALTH_CHECK_RESULT=!ERRORLEVEL!
echo DEBUG: Health check returned: !HEALTH_CHECK_RESULT!

if "!HEALTH_CHECK_RESULT!"=="0" (
    echo DEBUG: Health check passed, continuing...
    goto health_check_passed
)

echo.
echo ========================================
echo HEALTH CHECK FAILED
echo ========================================
echo.
echo Some issues were detected. Do you want to continue anyway? (Y/N)
set /p CONTINUE=
if /i not "!CONTINUE!"=="Y" (
    echo.
    echo Startup cancelled. Please fix the issues and try again.
    pause
    exit /b 1
)

:health_check_passed

echo.
echo ========================================
echo Health check passed! Starting server...
echo ========================================
echo.

REM Initialize database
echo Initializing database...
python init_database.py
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
) else (
    echo No virtual environment found at venv\
    echo.
    echo Do you want to create one? (Y/N)
    set /p CREATE_VENV=
    if /i "%CREATE_VENV%"=="Y" (
        echo.
        echo Checking for existing venv directory...
        if exist "venv" (
            echo WARNING: Found incomplete venv directory. Removing it...
            rmdir /s /q venv
            timeout /t 2 /nobreak >nul
        )
        echo Creating virtual environment...
        python -m venv venv
        if errorlevel 1 (
            echo ERROR: Failed to create virtual environment
            echo Continuing without venv...
            echo.
        ) else (
            echo Virtual environment created successfully
            call venv\Scripts\activate.bat
            echo.
            echo Installing requirements...
            echo This may take a few minutes...
            pip install --upgrade pip
            pip install -r requirements.txt
            echo.
        )
    ) else (
        echo Continuing without virtual environment...
        echo.
    )
)

REM Set default environment variables
if "%HOST%"=="" set HOST=0.0.0.0
if "%PORT%"=="" set PORT=5000
if "%DEBUG%"=="" set DEBUG=False

echo Configuration:
echo   Host: %HOST%
echo   Port: %PORT%
echo   Debug: %DEBUG%
echo.

REM Check if models exist
if not exist "utils\models\best.pt" (
    echo WARNING: Model file not found at utils\models\best.pt
    echo Plate detection may not work properly
    echo.
)

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "captures" mkdir captures
if not exist "templates" mkdir templates
if not exist "instance" mkdir instance

echo Starting web server...
echo.
echo Access the dashboard at: http://localhost:%PORT%
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the Flask app
python app.py

REM If server stops
echo.
echo Server stopped.
pause
