"""
Speed Violation Detector for live detection
"""
import os
import cv2
import math
import numpy as np
from ultralytics import YOLO
from config import Config

class SpeedDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.seen_obj_ids_speed = set()
        self.violations = []
        self.last_positions = {}
        self.speeds = {}
        self.snapshot_folder = Config.SNAPSHOT_FOLDER
        os.makedirs(self.snapshot_folder, exist_ok=True)
        self.calibration = None
    
    def initialize(self, calibration_data):
        """Initialize detector with calibration data"""
        if calibration_data is None:
            raise ValueError("Speed detection requires calibration data")
        
        self.calibration = calibration_data
        self.PIXELS_PER_METER = float(calibration_data.get("pixels_per_meter", 1.0))
        self.lines = calibration_data.get("lines", [])
        self.frame_time = float(calibration_data.get("frame_time", 0.033))
        
        if len(self.lines) < 2:
            raise ValueError("Calibration must provide 2 parallel lines")
        
        self.line1, self.line2 = self.lines
        self.seen_obj_ids_speed.clear()
        self.violations.clear()
        self.last_positions.clear()
        self.speeds.clear()
    
    def detect(self, frame, tracked_vehicles, frame_idx, original_frame):
        """
        Detect speed violations
        
        Args:
            frame: Frame to draw on
            tracked_vehicles: List of tracked vehicles
            frame_idx: Current frame index
            original_frame: Original frame
        
        Returns:
            List of violations found in this frame
        """
        violations_found = []
        
        for vehicle in tracked_vehicles:
            obj_id = vehicle.get('id')
            bbox = vehicle.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Calculate speed
            speed = self._calculate_speed(
                obj_id, cx, cy, frame, bbox, original_frame, frame_idx
            )
            
            # Check for speed violation (threshold: 30 km/h)
            if speed and speed > 30:
                if int(obj_id) not in self.seen_obj_ids_speed:
                    snap_name = f"speed_{obj_id}_{int(cv2.getTickCount())}.jpg"
                    snap_full_path = os.path.join(self.snapshot_folder, snap_name)
                    
                    crop = original_frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(snap_full_path, crop)
                    
                    violation = {
                        "type": "speed",
                        "confidence": 0.9,  # High confidence for speed violations
                        "bbox": [x1, y1, x2, y2],
                        "snapshot_name": snap_name,
                        "vehicle_id": int(obj_id),
                        "frame": frame_idx,
                        "timestamp": frame_idx * self.frame_time,
                        "metadata": {
                            "speed": speed
                        },
                        "snapshot_url": f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                    }
                    
                    violations_found.append(violation)
                    self.violations.append(violation)
                    self.seen_obj_ids_speed.add(int(obj_id))
        
        return violations_found
    
    def _calculate_speed(self, obj_id, cx, cy, frame, box, original_frame, frame_idx):
        """Calculate speed for a vehicle"""
        if not self._is_inside_lines((cx, cy)):
            self.last_positions[obj_id] = (cx, cy)
            return None
        
        if obj_id in self.last_positions:
            last_cx, last_cy = self.last_positions[obj_id]
            dx, dy = cx - last_cx, cy - last_cy
            pixel_dist = math.sqrt(dx**2 + dy**2)
            dist_m = pixel_dist / self.PIXELS_PER_METER
            speed = (dist_m / self.frame_time) * 3.6  # km/h
            
            # Smooth speed calculation
            self.speeds[obj_id] = 0.8 * self.speeds.get(obj_id, speed) + 0.2 * speed
            
            # Draw bounding box and speed
            x1, y1, x2, y2 = [int(v) for v in box]
            current_speed = self.speeds[obj_id]
            
            color = (255, 0, 0) if current_speed > 30 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{current_speed:.1f} km/h", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 0), 2)
            
            return current_speed
        
        self.last_positions[obj_id] = (cx, cy)
        return None
    
    def _is_inside_lines(self, p):
        """Check if point is between two parallel lines"""
        p = np.array(p)
        a1, b1 = np.array(self.line1[0]), np.array(self.line1[1])
        a2 = np.array(self.line2[0])
        
        v = b1 - a1
        v_norm = v / np.linalg.norm(v)
        v_perp = np.array([-v_norm[1], v_norm[0]])
        dist = np.dot(p - a1, v_perp)
        line_dist = np.dot(a2 - a1, v_perp)
        
        return 0 <= dist <= line_dist if line_dist > 0 else line_dist <= dist <= 0
    
    def get_violations(self):
        """Get all detected violations"""
        return self.violations.copy()
    
    def reset(self):
        """Reset detector state"""
        self.seen_obj_ids_speed.clear()
        self.violations.clear()
        self.last_positions.clear()
        self.speeds.clear()

