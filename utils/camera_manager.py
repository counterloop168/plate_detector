"""
Thread-Safe Camera Manager
Handles camera lifecycle and prevents race conditions
"""

import cv2
import logging
from threading import RLock
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CameraManager:
    """Thread-safe manager for camera resources"""
    
    def __init__(self):
        self._cameras = {}
        self._locks = {}
        self._global_lock = RLock()
    
    def add_camera(self, camera_id, camera_source=0):
        """
        Add a camera to the manager
        
        Args:
            camera_id: Camera identifier
            camera_source: Camera source (0 for default, or URL for IP camera)
        
        Returns:
            cv2.VideoCapture or None if failed
        """
        with self._global_lock:
            # Check if camera already exists
            if camera_id in self._cameras and self._cameras[camera_id] is not None:
                logger.warning(f"Camera {camera_id} already active")
                return self._cameras[camera_id]
            
            # Create camera capture
            logger.info(f"Initializing camera {camera_id} with source: {camera_source}")
            cap = cv2.VideoCapture(camera_source)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return None
            
            # Set camera properties
            from utils.constants import CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT, CAMERA_FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
            # Store camera
            self._cameras[camera_id] = cap
            self._locks[camera_id] = RLock()
            
            logger.info(f"Camera {camera_id} initialized successfully")
            return cap
    
    def get_camera(self, camera_id):
        """
        Get camera capture object
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            cv2.VideoCapture or None
        """
        with self._global_lock:
            return self._cameras.get(camera_id)
    
    @contextmanager
    def acquire_camera(self, camera_id):
        """
        Context manager to safely acquire camera lock
        
        Usage:
            with camera_manager.acquire_camera(1) as camera:
                if camera:
                    ret, frame = camera.read()
        
        Args:
            camera_id: Camera identifier
        
        Yields:
            cv2.VideoCapture or None
        """
        # Get camera-specific lock
        with self._global_lock:
            if camera_id not in self._locks:
                self._locks[camera_id] = RLock()
            lock = self._locks[camera_id]
            camera = self._cameras.get(camera_id)
        
        # Acquire camera lock
        with lock:
            yield camera
    
    def remove_camera(self, camera_id):
        """
        Remove and release camera
        
        Args:
            camera_id: Camera identifier
        """
        with self._global_lock:
            if camera_id in self._cameras:
                camera = self._cameras[camera_id]
                if camera is not None:
                    logger.info(f"Releasing camera {camera_id}")
                    try:
                        camera.release()
                    except Exception as e:
                        logger.error(f"Error releasing camera {camera_id}: {e}")
                    self._cameras[camera_id] = None
                
                # Remove from dict
                del self._cameras[camera_id]
                
                # Clean up lock
                if camera_id in self._locks:
                    del self._locks[camera_id]
    
    def is_active(self, camera_id):
        """
        Check if camera is active
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            bool: True if camera is active
        """
        with self._global_lock:
            return (camera_id in self._cameras and 
                    self._cameras[camera_id] is not None and
                    self._cameras[camera_id].isOpened())
    
    def get_active_cameras(self):
        """
        Get list of active camera IDs
        
        Returns:
            list: List of active camera IDs
        """
        with self._global_lock:
            return [cid for cid, cam in self._cameras.items() 
                    if cam is not None and cam.isOpened()]
    
    def release_all(self):
        """Release all cameras"""
        with self._global_lock:
            camera_ids = list(self._cameras.keys())
            for camera_id in camera_ids:
                self.remove_camera(camera_id)
            
            logger.info("All cameras released")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.release_all()
        except:
            pass


# Global camera manager instance
camera_manager = CameraManager()
