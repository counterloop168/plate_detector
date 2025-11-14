"""
Plate Detection Persistence Module
Handles storage, Google Drive uploads, and Google Sheets logging
"""

import os
import cv2
import json
import sqlite3
import logging
import gspread
from datetime import datetime, date, timedelta
from threading import Lock
from oauth2client.service_account import ServiceAccountCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'plate_detection_config.json')
CREDS_FILE = os.path.join(os.path.dirname(__file__), 'plate_detection_credentials.json')
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instance', 'site.db')

# Global instances
gauth = None
drive = None
sheets_client = None
persistence_lock = Lock()

# Configuration
config = {}


def load_config():
    """Load persistence configuration"""
    global config
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return True
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return False


def init_google_drive():
    """Initialize Google Drive client"""
    global gauth, drive
    
    try:
        if not config.get('google_drive', {}).get('enabled', False):
            logger.info("Google Drive uploads disabled")
            return False
        
        gauth = GoogleAuth()
        drive = GoogleDrive(gauth)
        logger.info("Google Drive initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Google Drive initialization failed: {e}")
        return False


def init_google_sheets():
    """Initialize Google Sheets client"""
    global sheets_client
    
    try:
        if not config.get('google_sheets', {}).get('enabled', False):
            logger.info("Google Sheets integration disabled")
            return False
        
        if not os.path.exists(CREDS_FILE):
            logger.error(f"Credentials file not found: {CREDS_FILE}")
            return False
        
        scopes = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scopes)
        sheets_client = gspread.authorize(creds)
        
        logger.info("Google Sheets initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Google Sheets initialization failed: {e}")
        return False


def get_or_create_drive_folder(parent_folder_id, folder_name):
    """Get or create a folder in Google Drive"""
    if drive is None:
        return None
    
    try:
        # Check if folder exists
        query = f"'{parent_folder_id}' in parents and title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        folder_list = drive.ListFile({'q': query}).GetList()
        
        if folder_list:
            return folder_list[0]['id']
        
        # Create new folder
        folder = drive.CreateFile({
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [{'id': parent_folder_id}]
        })
        folder.Upload()
        logger.info(f"Created Google Drive folder: {folder_name}")
        return folder['id']
        
    except Exception as e:
        logger.error(f"Error creating Google Drive folder {folder_name}: {e}")
        return None


def upload_to_google_drive(file_path, parent_folder_id):
    """Upload file to Google Drive and return link"""
    if drive is None:
        return None, None
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found for upload: {file_path}")
            return None, None
        
        file_name = os.path.basename(file_path)
        
        # Upload file
        gfile = drive.CreateFile({
            'parents': [{'id': parent_folder_id}],
            'title': file_name
        })
        gfile.SetContentFile(file_path)
        gfile.Upload()
        
        # Make publicly readable
        try:
            gfile.InsertPermission({'type': 'anyone', 'value': '', 'role': 'reader'})
        except:
            pass
        
        file_id = gfile.get('id')
        image_link = f"https://drive.google.com/uc?export=view&id={file_id}"
        folder_link = f"https://drive.google.com/drive/folders/{parent_folder_id}"
        
        logger.debug(f"Uploaded to Google Drive: {file_name}, id={file_id}")
        return image_link, folder_link
        
    except Exception as e:
        logger.error(f"Error uploading {file_path} to Google Drive: {e}")
        return None, None


def save_plate_image_local(frame, camera_id, plate_number, direction):
    """
    Save plate detection image locally
    
    Args:
        frame: Image frame (numpy array)
        camera_id: Camera identifier
        plate_number: Detected plate number
        direction: 'IN' or 'OUT'
    
    Returns:
        local_path: Path to saved image
    """
    try:
        # Create directory structure: captures/YYYY-MM-DD/camera_{id}_{direction}/plate_detections/
        today_date = datetime.now().strftime("%Y-%m-%d")
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'captures')
        date_dir = os.path.join(base_dir, today_date)
        camera_dir = os.path.join(date_dir, f"camera_{camera_id}_{direction.lower()}")
        plate_dir = os.path.join(camera_dir, 'plate_detections')
        
        os.makedirs(plate_dir, exist_ok=True)
        
        # Generate filename: HHMMSS_plate123.jpg
        timestamp_str = datetime.now().strftime("%H%M%S")
        safe_plate = plate_number.replace('-', '').replace(' ', '')
        filename = f"{timestamp_str}_{safe_plate}.jpg"
        local_path = os.path.join(plate_dir, filename)
        
        # Save image
        cv2.imwrite(local_path, frame)
        logger.debug(f"Saved plate image locally: {local_path}")
        
        return local_path
        
    except Exception as e:
        logger.error(f"Error saving plate image locally: {e}")
        return None


def upload_plate_image_to_drive(local_path, camera_id, direction):
    """
    Upload plate image to Google Drive
    
    Args:
        local_path: Path to local image file
        camera_id: Camera identifier
        direction: 'IN' or 'OUT'
    
    Returns:
        (image_link, folder_link): Tuple of image and folder links
    """
    if drive is None:
        return None, None
    
    try:
        # Create folder structure: Base/Plate_Detections/YYYY-MM-DD/IN or OUT/
        base_folder_id = config.get('google_drive', {}).get('base_folder_id')
        if not base_folder_id:
            logger.error("Base folder ID not configured")
            return None, None
        
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get or create Plate_Detections folder
        plate_folder_id = get_or_create_drive_folder(base_folder_id, 'Plate_Detections')
        if not plate_folder_id:
            return None, None
        
        # Get or create date folder
        date_folder_id = get_or_create_drive_folder(plate_folder_id, today_date)
        if not date_folder_id:
            return None, None
        
        # Get or create direction folder (IN/OUT)
        direction_folder_id = get_or_create_drive_folder(date_folder_id, direction.upper())
        if not direction_folder_id:
            return None, None
        
        # Upload image
        return upload_to_google_drive(local_path, direction_folder_id)
        
    except Exception as e:
        logger.error(f"Error uploading plate image to Drive: {e}")
        return None, None


def append_to_google_sheet(plate_data):
    """
    Append plate detection to Google Sheets
    
    Args:
        plate_data: Dict with keys: timestamp, camera_id, direction, plate_number, 
                    confidence, image_link, folder_link
    
    Returns:
        bool: Success status
    """
    if sheets_client is None:
        return False
    
    try:
        spreadsheet_id = config.get('google_sheets', {}).get('spreadsheet_id')
        if not spreadsheet_id:
            logger.error("Spreadsheet ID not configured")
            return False
        
        # Open spreadsheet
        spreadsheet = sheets_client.open_by_key(spreadsheet_id)
        
        # Get or create today's sheet
        today_date = datetime.now().strftime("%Y-%m-%d")
        try:
            worksheet = spreadsheet.worksheet(today_date)
        except gspread.WorksheetNotFound:
            # Create new sheet
            worksheet = spreadsheet.add_worksheet(title=today_date, rows=1000, cols=6)
            # Add header row
            headers = ['Timestamp', 'Camera', 'Plate Number', 'Confidence', 'Image Link (Drive)', 'Direction Folder Link']
            worksheet.append_row(headers)
        
        # Prepare row data
        row = [
            plate_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            f"Camera {plate_data.get('camera_id', '')}",
            plate_data.get('plate_number', ''),
            f"{plate_data.get('confidence', 0):.1f}%",
            plate_data.get('image_link', ''),
            plate_data.get('folder_link', '')
        ]
        
        # Append row
        worksheet.append_row(row)
        logger.info(f"Appended to Google Sheet: {plate_data.get('plate_number')}")
        return True
        
    except Exception as e:
        logger.error(f"Error appending to Google Sheet: {e}")
        return False


def save_plate_detection_to_db(camera_id, plate_number, confidence, direction, 
                                 image_path, drive_link, drive_folder_link, 
                                 tracking_id, is_duplicate=False):
    """
    Save plate detection to database
    
    Args:
        camera_id: Camera identifier
        plate_number: Detected plate text
        confidence: Detection confidence (0-100)
        direction: 'IN' or 'OUT'
        image_path: Local image path
        drive_link: Google Drive image link
        drive_folder_link: Google Drive folder link
        tracking_id: Tracking ID
        is_duplicate: Whether this is flagged as duplicate
    
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert into plate_detection table
        cursor.execute("""
            INSERT INTO plate_detection 
            (plate_number, confidence, timestamp, camera_id, direction, 
             image_path, drive_link, drive_folder_link, is_duplicate, tracking_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            plate_number,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            camera_id,
            direction,
            image_path or '',
            drive_link or '',
            drive_folder_link or '',
            1 if is_duplicate else 0,
            str(tracking_id) if tracking_id else ''
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved plate detection to DB: {plate_number} (Camera {camera_id}, {direction})")
        return True
        
    except Exception as e:
        logger.error(f"Error saving plate detection to DB: {e}")
        return False


def persist_plate_detection(frame, camera_id, plate_number, confidence, 
                             direction, tracking_id, is_duplicate=False):
    """
    Complete persistence workflow for a plate detection
    
    Args:
        frame: Image frame
        camera_id: Camera identifier
        plate_number: Detected plate text
        confidence: Detection confidence (0-100)
        direction: 'IN' or 'OUT'
        tracking_id: Tracking ID
        is_duplicate: Whether flagged as duplicate
    
    Returns:
        dict: Status of each persistence operation
    """
    result = {
        'local_save': False,
        'drive_upload': False,
        'sheets_append': False,
        'db_save': False,
        'local_path': None,
        'drive_link': None,
        'folder_link': None
    }
    
    try:
        # Save locally
        local_path = save_plate_image_local(frame, camera_id, plate_number, direction)
        if local_path:
            result['local_save'] = True
            result['local_path'] = local_path
        
        # Upload to Drive
        if drive is not None and local_path:
            image_link, folder_link = upload_plate_image_to_drive(local_path, camera_id, direction)
            if image_link:
                result['drive_upload'] = True
                result['drive_link'] = image_link
                result['folder_link'] = folder_link
        
        # Append to Sheets
        if sheets_client is not None and not is_duplicate:
            plate_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'camera_id': camera_id,
                'direction': direction,
                'plate_number': plate_number,
                'confidence': confidence,
                'image_link': result.get('drive_link', ''),
                'folder_link': result.get('folder_link', '')
            }
            if append_to_google_sheet(plate_data):
                result['sheets_append'] = True
        
        # Save to database
        if save_plate_detection_to_db(
            camera_id, plate_number, confidence, direction,
            result.get('local_path'), result.get('drive_link'), 
            result.get('folder_link'), tracking_id, is_duplicate
        ):
            result['db_save'] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Error in persist_plate_detection: {e}")
        return result


def cleanup_old_files(days=7):
    """
    Cleanup local files older than specified days
    
    Args:
        days: Number of days to retain files
    
    Returns:
        int: Number of files deleted
    """
    try:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'captures')
        if not os.path.exists(base_dir):
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        # Iterate through date folders
        for date_folder in os.listdir(base_dir):
            date_path = os.path.join(base_dir, date_folder)
            if not os.path.isdir(date_path):
                continue
            
            try:
                # Parse date from folder name
                folder_date = datetime.strptime(date_folder, "%Y-%m-%d")
                
                if folder_date < cutoff_date:
                    # Delete entire date folder
                    import shutil
                    shutil.rmtree(date_path)
                    deleted_count += 1
                    logger.info(f"Deleted old folder: {date_path}")
            except ValueError:
                # Skip folders that don't match date format
                continue
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return 0


def cleanup_old_drive_folders(days=7):
    """
    Cleanup Google Drive folders older than specified days
    
    Args:
        days: Number of days to retain folders
    
    Returns:
        int: Number of folders deleted
    """
    if drive is None:
        return 0
    
    try:
        base_folder_id = config.get('google_drive', {}).get('base_folder_id')
        if not base_folder_id:
            return 0
        
        # Get Plate_Detections folder
        query = f"'{base_folder_id}' in parents and title='Plate_Detections' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        folder_list = drive.ListFile({'q': query}).GetList()
        
        if not folder_list:
            return 0
        
        plate_folder_id = folder_list[0]['id']
        
        # Get all date folders
        query = f"'{plate_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        date_folders = drive.ListFile({'q': query}).GetList()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for folder in date_folders:
            try:
                # Parse date from folder name
                folder_date = datetime.strptime(folder['title'], "%Y-%m-%d")
                
                if folder_date < cutoff_date:
                    # Delete folder
                    folder.Delete()
                    deleted_count += 1
                    logger.info(f"Deleted old Drive folder: {folder['title']}")
            except ValueError:
                # Skip folders that don't match date format
                continue
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error during Drive cleanup: {e}")
        return 0


def cleanup_old_sheets(days=7):
    """
    Cleanup Google Sheets older than specified days
    
    Args:
        days: Number of days to retain sheets
    
    Returns:
        int: Number of sheets deleted
    """
    if sheets_client is None:
        return 0
    
    try:
        spreadsheet_id = config.get('google_sheets', {}).get('spreadsheet_id')
        if not spreadsheet_id:
            return 0
        
        spreadsheet = sheets_client.open_by_key(spreadsheet_id)
        worksheets = spreadsheet.worksheets()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for worksheet in worksheets:
            try:
                # Parse date from sheet name
                sheet_date = datetime.strptime(worksheet.title, "%Y-%m-%d")
                
                if sheet_date < cutoff_date:
                    # Delete sheet
                    spreadsheet.del_worksheet(worksheet)
                    deleted_count += 1
                    logger.info(f"Deleted old sheet: {worksheet.title}")
            except ValueError:
                # Skip sheets that don't match date format
                continue
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error during Sheets cleanup: {e}")
        return 0


# Initialize on module load
load_config()
init_google_drive()
init_google_sheets()

logger.info("Plate persistence module loaded")
