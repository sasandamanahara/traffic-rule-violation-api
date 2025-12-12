"""
No Parking Violation Detector for live detection
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from config import Config

class NoParkingDetector:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
        self.sign_model = YOLO(os.path.join(MODELS_DIR, 'Parking_best.pt'))
        self.vehicle_model = YOLO("yolov8n.pt")
        self.seen_obj_ids_noparking = set()
        self.violations = []
        self.snapshot_folder = Config.SNAPSHOT_FOLDER
        os.makedirs(self.snapshot_folder, exist_ok=True)
        
        self.near_count = {}  # vehicle_id -> frames near the sign
        self.VIOLATION_FRAMES = 200  # Frames threshold for violation
        self.DIST_THRESHOLD = 250  # Distance threshold in pixels
        self.tracker = {}  # Simple tracker
        self.next_id = 0
        self.sign_centers = []
        self.sign_detected = False
    
    def initialize(self, calibration_data=None):
        """Initialize detector (no calibration needed)"""
        self.seen_obj_ids_noparking.clear()
        self.violations.clear()
        self.near_count.clear()
        self.tracker.clear()
        self.next_id = 0
        self.sign_centers = []
        self.sign_detected = False
    
    def detect(self, frame, tracked_vehicles, frame_idx, original_frame):
        """
        Detect no parking violations
        
        Args:
            frame: Frame to draw on
            tracked_vehicles: List of tracked vehicles
            frame_idx: Current frame index
            original_frame: Original frame
        
        Returns:
            List of violations found in this frame
        """
        violations_found = []
        
        # Detect no parking signs (once per session or periodically)
        if frame_idx == 1 or frame_idx % 300 == 0:  # Re-detect every 300 frames
            self._detect_parking_signs(original_frame)
        
        # Process vehicles
        for vehicle in tracked_vehicles:
            obj_id = vehicle.get('id')
            bbox = vehicle.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Distance to nearest no-parking sign
            min_dist = 9999
            for (scx, scy) in self.sign_centers:
                dist = np.sqrt((cx - scx) ** 2 + (cy - scy) ** 2)
                min_dist = min(min_dist, dist)
            
            # Violation check
            if min_dist < self.DIST_THRESHOLD:
                self.near_count[obj_id] = self.near_count.get(obj_id, 0) + 1
            else:
                self.near_count[obj_id] = 0
            
            # Check if violation threshold reached
            if self.near_count.get(obj_id, 0) >= self.VIOLATION_FRAMES:
                if int(obj_id) not in self.seen_obj_ids_noparking:
                    snap_name = f"noparking_{obj_id}_{int(cv2.getTickCount())}.jpg"
                    snap_full_path = os.path.join(self.snapshot_folder, snap_name)
                    
                    cv2.imwrite(snap_full_path, original_frame)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "PARKING VIOLATION!", (x1, y1 - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    violation_data = {
                        "type": "no_parking",
                        "confidence": 0.8,
                        "bbox": [x1, y1, x2, y2],
                        "snapshot_name": snap_name,
                        "vehicle_id": int(obj_id) if obj_id is not None else None,
                        "frame": frame_idx,
                        "timestamp": frame_idx / 30.0,
                        "snapshot_url": f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                    }
                    
                    violations_found.append(violation_data)
                    self.violations.append(violation_data)
                    self.seen_obj_ids_noparking.add(int(obj_id) if obj_id is not None else 0)
        
        # Draw no parking zones
        overlay = frame.copy()
        for (scx, scy) in self.sign_centers:
            cv2.circle(overlay, (scx, scy), self.DIST_THRESHOLD, (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        return violations_found
    
    def _detect_parking_signs(self, frame):
        """Detect no parking signs in the frame"""
        self.sign_centers = []
        
        # Resize frame for faster processing (optional)
        resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
        
        sign_results = self.sign_model(resized, verbose=False)
        
        for r in sign_results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                # Scale back to original size
                x1, y1, x2, y2 = x1 * 2, y1 * 2, x2 * 2, y2 * 2
                
                # Store sign center (below the sign)
                self.sign_centers.append((x1, y2 + 100))
                self.sign_detected = True
                
                # Draw sign box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "No Parking", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def get_violations(self):
        """Get all detected violations"""
        return self.violations.copy()
    
    def reset(self):
        """Reset detector state"""
        self.seen_obj_ids_noparking.clear()
        self.violations.clear()
        self.near_count.clear()
        self.tracker.clear()
        self.next_id = 0
        self.sign_centers = []
        self.sign_detected = False


