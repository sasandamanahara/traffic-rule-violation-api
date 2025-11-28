"""
Live Detection Service
Manages detection state, frame queue for streaming, and violation events
"""
import threading
import queue
import time
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
from collections import deque
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db_models import ViolationModel
from config import Config


def map_violation_type(live_type):
    """
    Map live detection violation types to database format
    
    Args:
        live_type (str): Violation type from live detection (e.g., "speed", "helmet")
        
    Returns:
        str: Database violation type (e.g., "Speed Limit", "Helmet")
    """
    type_mapping = {
        'speed': 'Speed Limit',
        'direction': 'Wrong Direction',
        'helmet': 'Helmet',
        'triple_riding': 'Triple Riding',
        'red_light': 'Red Light'
    }
    return type_mapping.get(live_type.lower(), live_type.title())


class LiveDetectionService:
    """Service to manage live detection state and data queues"""
    
    def __init__(self, max_frame_queue_size=10, max_violations=100):
        self.is_running = False
        self.detection_thread = None
        self.video_source = None
        self.calibration = None
        self.vehicle_directions = None
        
        # Frame queue for MJPEG streaming
        self.frame_queue = queue.Queue(maxsize=max_frame_queue_size)
        
        # Violations storage (most recent first)
        self.violations = deque(maxlen=max_violations)
        self.violations_lock = threading.Lock()
        
        # Status tracking
        self.status = {
            'running': False,
            'connected': False,
            'frame_count': 0,
            'violation_count': 0,
            'start_time': None,
            'error': None
        }
        self.status_lock = threading.Lock()
        
    def start_detection(self, video_source, calibration, vehicle_directions):
        """Start live detection in a background thread"""
        if self.is_running:
            return {'error': 'Detection already running'}
        
        self.video_source = video_source
        self.calibration = calibration
        self.vehicle_directions = vehicle_directions
        
        self.is_running = True
        self.detection_thread = threading.Thread(
            target=self._run_detection,
            daemon=True
        )
        self.detection_thread.start()
        
        return {'success': True, 'message': 'Detection started'}
    
    def stop_detection(self):
        """Stop live detection"""
        if not self.is_running:
            return {'error': 'Detection not running'}
        
        self.is_running = False
        
        # Wait for thread to finish (with timeout)
        if self.detection_thread:
            self.detection_thread.join(timeout=5.0)
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        with self.status_lock:
            self.status['running'] = False
            self.status['connected'] = False
        
        return {'success': True, 'message': 'Detection stopped'}
    
    def get_latest_frame(self):
        """Get the latest frame from queue (for MJPEG streaming)"""
        try:
            # Get the most recent frame, discarding older ones
            frame = None
            while True:
                try:
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            return frame
        except Exception as e:
            print(f"[ERROR] Error getting frame: {e}")
            return None
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types"""
        import numpy as np
        
        # Handle None first
        if obj is None:
            return None
        
        # Handle numpy scalars - use item() to convert to Python native type
        if isinstance(obj, np.generic):
            return obj.item()
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle standard Python types
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle collections
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        
        # Fallback
        try:
            if hasattr(obj, 'item'):
                return obj.item()
            return obj
        except:
            return obj
    
    def add_violation(self, violation_type, vehicle_id, frame, bbox=None, metadata=None):
        """Add a violation to the violations list and save to database"""
        with self.violations_lock:
            violation_id = len(self.violations) + 1
        
        # Convert numpy types in bbox and metadata to Python native types
        converted_bbox = self._convert_numpy_types(bbox) if bbox else None
        converted_metadata = self._convert_numpy_types(metadata) if metadata else {}
        converted_vehicle_id = self._convert_numpy_types(vehicle_id) if vehicle_id is not None else None
        
        violation = {
            'id': violation_id,
            'type': violation_type,
            'vehicle_id': converted_vehicle_id,
            'timestamp': datetime.now().isoformat(),
            'bbox': converted_bbox,
            'metadata': converted_metadata
        }
        
        # Encode frame to JPEG for storage
        if frame is not None:
            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                violation['image_data'] = buffer.tobytes()
            except Exception as e:
                print(f"[ERROR] Error encoding violation image: {e}")
        
        with self.violations_lock:
            self.violations.appendleft(violation)  # Most recent first
            violation_count = len(self.violations)
            # print(f"[DEBUG] Added violation: {violation_type}, ID: {violation_id}, Total: {violation_count}")
        
        with self.status_lock:
            self.status['violation_count'] = violation_count
        
        # Save to database (outside lock to avoid blocking)
        self._save_violation_to_db(violation_type, vehicle_id, frame, bbox, metadata)
    
    def _save_violation_to_db(self, violation_type, vehicle_id, frame, bbox, metadata):
        """
        Save violation to database and save snapshot image
        
        Args:
            violation_type (str): Type of violation (e.g., "speed", "helmet")
            vehicle_id: Vehicle tracking ID
            frame: Violation frame image (numpy array)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            metadata: Additional violation metadata
        """
        try:
            # Ensure snapshot folder exists
            os.makedirs(Config.SNAPSHOT_FOLDER, exist_ok=True)
            
            # Save violation image to snapshot folder
            snapshot_filename = None
            snapshot_url = None
            
            if frame is not None:
                try:
                    snapshot_filename = f"{uuid.uuid4()}.jpg"
                    snapshot_path = os.path.join(Config.SNAPSHOT_FOLDER, snapshot_filename)
                    cv2.imwrite(snapshot_path, frame)
                    snapshot_url = f"{Config.API_BASE_URL}/static/snapshots/{snapshot_filename}"
                except Exception as e:
                    print(f"[ERROR] Error saving violation snapshot: {e}")
            
            # Map violation type to database format
            db_violation_type = map_violation_type(violation_type)
            
            # Convert timestamp from ISO string to float (seconds since start)
            # For live detection, we'll use current time as timestamp
            violation_timestamp = datetime.now()
            timestamp_seconds = violation_timestamp.timestamp()
            
            # Build description from violation type and metadata
            description_parts = [f"{db_violation_type} violation detected"]
            if metadata:
                if 'speed' in metadata:
                    description_parts.append(f"Speed: {metadata['speed']:.1f} km/h")
            description = ". ".join(description_parts)
            
            # Prepare database record
            db_violation = {
                'type': db_violation_type,
                'vehicle_id': str(vehicle_id) if vehicle_id is not None else None,
                'timestamp': timestamp_seconds,
                'bbox': bbox if bbox else [],
                'snapshot_url': snapshot_url,
                'video_file': self.video_source if self.video_source else 'Live Stream',  # RTSP URL or file path
                'status': 'pending',
                'location': 'Live Detection',
                'description': description,
                'metadata': metadata or {}
            }
            
            # Add confidence if available in metadata
            if metadata and 'confidence' in metadata:
                db_violation['confidence'] = metadata['confidence']
            
            # Save to database
            violation_id = ViolationModel.create_violation(db_violation)
            if violation_id:
                print(f"[INFO] Saved violation to database: {db_violation_type} (ID: {violation_id})")
            else:
                print(f"[WARN] Failed to save violation to database (MongoDB may be unavailable)")
                
        except Exception as e:
            # Log error but don't break live detection
            print(f"[ERROR] Error saving violation to database: {e}")
            import traceback
            traceback.print_exc()
    
    def get_violations(self, limit=50):
        """Get recent violations"""
        with self.violations_lock:
            violations_list = list(self.violations)[:limit]
            # print(f"[DEBUG] get_violations: returning {len(violations_list)} violations (total in deque: {len(self.violations)})")
            return violations_list
    
    def get_status(self):
        """Get current detection status"""
        with self.status_lock:
            return self.status.copy()
    
    def _run_detection(self):
        """Main detection loop (runs in background thread)"""
        from vehicle_detection import detect_vehicles_streaming
        
        with self.status_lock:
            self.status['running'] = True
            self.status['start_time'] = datetime.now().isoformat()
            self.status['error'] = None
        
        try:
            detect_vehicles_streaming(
                self.video_source,
                self.calibration,
                self.vehicle_directions,
                self.frame_queue,
                self.add_violation,
                lambda: self.is_running,
                self._update_status
            )
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Detection error: {error_msg}")
            with self.status_lock:
                self.status['error'] = error_msg
        finally:
            with self.status_lock:
                self.status['running'] = False
                self.status['connected'] = False
            self.is_running = False
    
    def _update_status(self, **kwargs):
        """Update status fields"""
        with self.status_lock:
            self.status.update(kwargs)


# Global service instance
_detection_service = None
_service_lock = threading.Lock()


def get_detection_service():
    """Get or create the global detection service instance"""
    global _detection_service
    with _service_lock:
        if _detection_service is None:
            _detection_service = LiveDetectionService()
        return _detection_service

