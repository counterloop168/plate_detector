"""
Database Initialization Script
Creates necessary tables and schema for plate detection system
"""

import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_DIR = os.path.join(os.path.dirname(__file__), 'instance')
DB_PATH = os.path.join(DB_DIR, 'site.db')


def init_database():
    """Initialize database with required tables"""
    try:
        # Create instance directory if not exists
        os.makedirs(DB_DIR, exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create plate_detection table
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
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate_timestamp 
            ON plate_detection(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate_camera 
            ON plate_detection(camera_id, timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate_number 
            ON plate_detection(plate_number)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database initialized successfully at %s", DB_PATH)
        logger.info("✅ Table 'plate_detection' created/verified")
        logger.info("✅ Indexes created for optimized queries")
        
        return True
        
    except sqlite3.Error as e:
        logger.error("❌ Database initialization failed: %s", str(e))
        return False


def verify_database():
    """Verify database structure"""
    try:
        if not os.path.exists(DB_PATH):
            logger.warning("⚠️  Database file does not exist: %s", DB_PATH)
            return False
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='plate_detection'
        """)
        
        if cursor.fetchone():
            # Get table info
            cursor.execute("PRAGMA table_info(plate_detection)")
            columns = cursor.fetchall()
            
            logger.info("✅ Database verified")
            logger.info("📊 Table 'plate_detection' has %d columns:", len(columns))
            for col in columns:
                logger.info("  - %s (%s)", col[1], col[2])
            
            # Get row count
            cursor.execute("SELECT COUNT(*) FROM plate_detection")
            count = cursor.fetchone()[0]
            logger.info("📈 Total records: %d", count)
            
            conn.close()
            return True
        else:
            logger.warning("⚠️  Table 'plate_detection' does not exist")
            conn.close()
            return False
            
    except sqlite3.Error as e:
        logger.error("❌ Database verification failed: %s", str(e))
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Plate Detection Database Initialization")
    print("=" * 60)
    print()
    
    # Initialize database
    if init_database():
        print()
        # Verify database
        verify_database()
        print()
        print("✅ Database setup completed successfully!")
    else:
        print()
        print("❌ Database setup failed!")
        print("Please check the error messages above")
    
    print()
