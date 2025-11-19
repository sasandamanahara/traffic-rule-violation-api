"""
Live Detection Service
Manages detection state, frame queue for streaming, and violation events
"""
import threading
import queue
import time
import cv2
import numpy as np
from datetime import datetime
from collections import deque


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
    
    def add_violation(self, violation_type, vehicle_id, frame, bbox=None, metadata=None):
        """Add a violation to the violations list"""
        violation = {
            'id': len(self.violations) + 1,
            'type': violation_type,
            'vehicle_id': vehicle_id,
            'timestamp': datetime.now().isoformat(),
            'bbox': bbox,
            'metadata': metadata or {}
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
        
        with self.status_lock:
            self.status['violation_count'] = len(self.violations)
    
    def get_violations(self, limit=50):
        """Get recent violations"""
        with self.violations_lock:
            return list(self.violations)[:limit]
    
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

