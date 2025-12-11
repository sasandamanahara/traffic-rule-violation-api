"""
Live Detection Service for real-time traffic violation detection
"""
import os
import cv2
import time
import threading
import queue
import uuid
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import numpy as np

from live_detectors import (
    HelmetTripleDetector,
    SpeedDetector,
    DirectionDetector,
    RedLightDetector,
    NoParkingDetector
)
from db_models import ViolationModel
from config import Config

# Import calibration functions
# Add the parent directory to path to allow importing from live/
try:
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    
    # Import initialize module
    from live.initialize import initialize_stream, RTSPStream
    
    # Import redlight_initialize
    from live.red_light_violation.redlight_initialize import detect_traffic_light_and_line
    
except ImportError as e:
    print(f"[ERROR] Could not import calibration functions: {str(e)}")
    print(f"[ERROR] Current sys.path: {sys.path[:3]}")
    print(f"[ERROR] Base directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("[ERROR] Some features may not work properly.")
    initialize_stream = None
    RTSPStream = None
    detect_traffic_light_and_line = None
except Exception as e:
    print(f"[ERROR] Unexpected error importing calibration functions: {str(e)}")
    import traceback
    traceback.print_exc()
    initialize_stream = None
    RTSPStream = None
    detect_traffic_light_and_line = None


class LiveDetectionService:
    """Main service for live camera detection"""
    
    def __init__(self):
        self.running = False
        self.connected = False
        self.video_source = None
        self.violation_types = []
        self.session_id = None
        
        # Thread management
        self.stream_thread = None
        self.detection_thread = None
        self.stopped = False
        
        # Frame queues
        self.frame_queue = queue.Queue(maxsize=5)
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        
        # Detection state
        self.frame_idx = 0
        self.violations = deque(maxlen=1000)  # Keep last 1000 violations in memory
        self.violation_lock = threading.Lock()
        
        # Models
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
        self.vehicle_model = YOLO(os.path.join(MODELS_DIR, 'new best.pt'))
        
        # Detectors
        self.detectors = {}
        self.calibration = None
        
        # RTSP stream
        self.stream = None
        self.cap = None
        
        # Frame processing settings
        self.frame_skip = 1  # Process every Nth frame (1 = all frames)
    
    def start(self, video_source, violation_types=None):
        """
        Start live detection
        
        Args:
            video_source: RTSP/RTMP URL or video file path
            violation_types: List of violation types to detect
                Options: ['helmet', 'triple_riding', 'speed', 'direction', 'red_light', 'no_parking']
        """
        if self.running:
            raise RuntimeError("Detection is already running")
        
        self.video_source = video_source
        self.violation_types = violation_types or ['helmet', 'triple_riding', 'speed', 'direction', 'red_light', 'no_parking']
        self.session_id = str(uuid.uuid4())
        self.stopped = False
        self.frame_idx = 0
        
        # Clear previous state
        with self.violation_lock:
            self.violations.clear()
        
        # Perform calibration
        print(f"[INFO] Starting calibration for {video_source}...")
        try:
            if initialize_stream is None:
                raise RuntimeError("Calibration functions not available. Check live/initialize.py imports.")
            calibration_result = initialize_stream(video_source)
            
            # Check for None or (None, None) failure cases
            if calibration_result is None:
                raise RuntimeError("Calibration failed - no calibration data returned")
            
            # Handle both old format (just calibration) and new format (calibration, stream)
            if isinstance(calibration_result, tuple) and len(calibration_result) == 2:
                self.calibration, self.stream = calibration_result
                # Check if calibration is None in tuple format
                if self.calibration is None:
                    raise RuntimeError("Calibration failed - no calibration data returned")
            else:
                # Backward compatibility with old format
                self.calibration = calibration_result
                self.stream = None
                if self.calibration is None:
                    raise RuntimeError("Calibration failed - no calibration data returned")
            
            # Validate calibration data
            if self.calibration is None:
                raise RuntimeError("Calibration failed - no calibration data returned")
            
            if not isinstance(self.calibration, dict):
                raise RuntimeError("Calibration data is not a valid dictionary")
            
            # Ensure all required fields exist with valid defaults
            required_fields = {
                "pixels_per_meter": 40.0,
                "lines": [
                    ((0, 240), (640, 240)),
                    ((0, 270), (640, 270))
                ],
                "frame_time": 0.033,
                "frame_size": (480, 640),
                "traffic_light_box": None,
                "traffic_light_line": None
            }
            
            for field, default_value in required_fields.items():
                if field not in self.calibration:
                    print(f"[WARN] Missing calibration field '{field}', using default")
                    self.calibration[field] = default_value
            
            # Validate pixels_per_meter
            if not isinstance(self.calibration["pixels_per_meter"], (int, float)) or self.calibration["pixels_per_meter"] <= 0:
                print(f"[WARN] Invalid pixels_per_meter: {self.calibration['pixels_per_meter']}, using default: 40.0")
                self.calibration["pixels_per_meter"] = 40.0
            
            # Validate lines
            if not isinstance(self.calibration["lines"], list) or len(self.calibration["lines"]) != 2:
                print(f"[WARN] Invalid lines in calibration, using defaults")
                h, w = self.calibration.get("frame_size", (480, 640))
                self.calibration["lines"] = [
                    ((0, h // 2 - 15), (w, h // 2 - 15)),
                    ((0, h // 2 + 15), (w, h // 2 + 15))
                ]
            
            # Validate frame_size
            if not isinstance(self.calibration["frame_size"], (tuple, list)) or len(self.calibration["frame_size"]) != 2:
                print(f"[WARN] Invalid frame_size, using default: (480, 640)")
                self.calibration["frame_size"] = (480, 640)
            
            print("[INFO] Calibration completed successfully")
        except Exception as e:
            print(f"[ERROR] Calibration error: {str(e)}")
            # Clean up stream if it was created
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.stop()
                except:
                    pass
            raise RuntimeError(f"Calibration failed: {str(e)}")
        
        # Initialize detectors
        self._initialize_detectors()
        
        # Start threads
        self.running = True
        self.connected = False
        
        # Start stream reader thread
        self.stream_thread = threading.Thread(target=self._stream_reader, daemon=True)
        self.stream_thread.start()
        
        # Wait a bit for stream to connect
        time.sleep(2)
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_processor, daemon=True)
        self.detection_thread.start()
        
        print(f"[INFO] Live detection started for {video_source}")
    
    def stop(self):
        """Stop live detection"""
        if not self.running:
            return
        
        print("[INFO] Stopping live detection...")
        self.stopped = True
        self.running = False
        self.connected = False
        
        # Stop stream
        if self.stream:
            try:
                self.stream.stop()
            except Exception as e:
                print(f"[WARN] Error stopping stream: {e}")
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[WARN] Error releasing video capture: {e}")
        
        # Wait for threads to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5)
        
        # Reset detectors
        for detector in self.detectors.values():
            detector.reset()
        
        print("[INFO] Live detection stopped")
    
    def get_status(self):
        """Get current detection status"""
        with self.violation_lock:
            violation_count = len(self.violations)
        
        return {
            "running": self.running,
            "connected": self.connected,
            "frame_count": self.frame_idx,
            "violation_count": violation_count,
            "video_source": self.video_source,
            "violation_types": self.violation_types,
            "session_id": self.session_id
        }
    
    def get_violations(self, limit=20):
        """Get recent violations"""
        with self.violation_lock:
            violations = list(self.violations)[-limit:]
        return violations
    
    def get_frame(self):
        """Get latest frame for streaming"""
        with self.latest_frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def _initialize_detectors(self):
        """Initialize detectors based on violation types"""
        self.detectors = {}
        
        if 'helmet' in self.violation_types or 'triple_riding' in self.violation_types:
            detector = HelmetTripleDetector()
            detector.initialize(self.calibration)
            self.detectors['helmet_triple'] = detector
        
        if 'speed' in self.violation_types:
            detector = SpeedDetector()
            detector.initialize(self.calibration)
            self.detectors['speed'] = detector
        
        if 'direction' in self.violation_types:
            detector = DirectionDetector()
            detector.initialize(self.calibration)
            self.detectors['direction'] = detector
        
        if 'red_light' in self.violation_types:
            detector = RedLightDetector()
            detector.initialize(self.calibration)
            self.detectors['red_light'] = detector
        
        if 'no_parking' in self.violation_types:
            detector = NoParkingDetector()
            detector.initialize(self.calibration)
            self.detectors['no_parking'] = detector
    
    def _stream_reader(self):
        """Thread for reading frames from stream"""
        print("[INFO] Stream reader thread started")
        
        # Determine if RTSP/RTMP or file
        is_stream = (self.video_source.startswith('rtsp://') or 
                    self.video_source.startswith('rtmp://'))
        
        if is_stream:
            # Reuse calibration stream if available, otherwise create new one
            if self.stream is None:
                if RTSPStream is None:
                    print("[ERROR] RTSPStream not available")
                    self.stopped = True
                    return
                print("[INFO] Creating new stream connection...")
                self.stream = RTSPStream(self.video_source)
            else:
                print("[INFO] Reusing calibration stream")
            
            # Wait for connection if not already connected
            max_wait = 30
            wait_count = 0
            while not self.stream.connected and not self.stream.connection_failed and not self.stopped and wait_count < max_wait:
                time.sleep(0.5)
                wait_count += 1
            
            if self.stopped:
                return
            
            if self.stream.connection_failed or not self.stream.connected:
                print("[ERROR] Stream connection failed in stream reader")
                self.connected = False
                self.stopped = True
                return
            
            self.connected = True
            print("[INFO] Stream connected")
            
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            while not self.stopped:
                ret, frame = self.stream.read()
                if ret and frame is not None:
                    consecutive_failures = 0
                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                    except queue.Full:
                        # Drop oldest frame if queue is full
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame, timeout=0.1)
                        except queue.Empty:
                            pass
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print("[WARN] Multiple consecutive frame read failures, stream may be disconnected")
                        self.connected = False
                        # Check if stream connection failed
                        if self.stream.connection_failed:
                            print("[ERROR] Stream connection permanently failed")
                            break
                    time.sleep(0.1)
        else:
            # File source
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                print(f"[ERROR] Cannot open video source: {self.video_source}")
                self.stopped = True
                return
            
            self.connected = True
            print("[INFO] Video file opened")
            
            while not self.stopped:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # End of video or error
                    break
                
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, timeout=0.1)
                    except queue.Empty:
                        pass
                
                time.sleep(0.033)  # ~30 FPS for file playback
        
        self.connected = False
        print("[INFO] Stream reader thread stopped")
    
    def _detection_processor(self):
        """Thread for processing frames and detecting violations"""
        print("[INFO] Detection processor thread started")
        
        while not self.stopped:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            # Skip frames if needed
            if self.frame_idx % self.frame_skip != 0:
                self.frame_idx += 1
                continue
            
            self.frame_idx += 1
            original_frame = frame.copy()
            
            # Run vehicle tracking
            try:
                results = self.vehicle_model.track(frame, persist=True, verbose=False)
            except Exception as e:
                print(f"[ERROR] Vehicle tracking failed: {str(e)}")
                continue
            
            # Extract tracked vehicles
            tracked_vehicles = []
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for box, obj_id, cls, conf in zip(boxes, ids, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    label = self.vehicle_model.names[int(cls)].lower()
                    
                    vehicle_type = 'motorcycle' if 'motor' in label or 'rider' in label else 'vehicle'
                    
                    tracked_vehicles.append({
                        'id': int(obj_id),
                        'bbox': [x1, y1, x2, y2],
                        'type': vehicle_type,
                        'class': int(cls),
                        'confidence': float(conf)
                    })
            
            # Run detectors
            frame_violations = []
            for detector_name, detector in self.detectors.items():
                try:
                    violations = detector.detect(
                        frame, tracked_vehicles, self.frame_idx, original_frame
                    )
                    frame_violations.extend(violations)
                except Exception as e:
                    print(f"[ERROR] Detector {detector_name} failed: {str(e)}")
                    continue
            
            # Store violations
            for violation in frame_violations:
                violation['session_id'] = self.session_id
                violation['live_detection'] = True
                violation['created_at'] = datetime.utcnow()
                
                # Save to database
                try:
                    violation_data = {
                        'type': violation.get('type', 'Unknown'),
                        'confidence': violation.get('confidence', 0.0),
                        'frame': violation.get('frame', 0),
                        'timestamp': violation.get('timestamp', 0),
                        'bbox': violation.get('bbox', [0, 0, 0, 0]),
                        'snapshot_url': violation.get('snapshot_url', ''),
                        'status': 'pending',
                        'location': 'Live Detection',
                        'description': f"Live detection: {violation.get('type', 'Unknown')} violation",
                        'live_detection': True,
                        'session_id': self.session_id,
                        'metadata': violation.get('metadata', {})
                    }
                    
                    ViolationModel.create_violation(violation_data)
                except Exception as e:
                    print(f"[ERROR] Failed to save violation to database: {str(e)}")
                
                # Store in memory
                with self.violation_lock:
                    self.violations.append(violation)
            
            # Update latest frame
            with self.latest_frame_lock:
                self.latest_frame = frame.copy()
        
        print("[INFO] Detection processor thread stopped")


# Global service instance
_service_instance = None
_service_lock = threading.Lock()


def get_service():
    """Get global service instance"""
    global _service_instance
    with _service_lock:
        if _service_instance is None:
            _service_instance = LiveDetectionService()
        return _service_instance

