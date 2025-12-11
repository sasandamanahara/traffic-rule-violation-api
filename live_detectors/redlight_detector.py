"""
Red Light Violation Detector for live detection
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from config import Config

class RedLightDetector:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
        self.model_vehicle = YOLO(os.path.join(MODELS_DIR, 'new best.pt'))
        self.seen_obj_ids_redlight = set()
        self.violations = []
        self.snapshot_folder = Config.SNAPSHOT_FOLDER
        os.makedirs(self.snapshot_folder, exist_ok=True)
        
        self.traffic_light_box = None
        self.violation_line_y = None
        self.light_history = []
        self.LIGHT_HISTORY_LEN = 5
        self.crossed_vehicles = set()
    
    def initialize(self, calibration_data):
        """Initialize detector with calibration data"""
        if calibration_data:
            self.traffic_light_box = calibration_data.get("traffic_light_box")
            self.violation_line_y = calibration_data.get("traffic_light_line")
        
        self.seen_obj_ids_redlight.clear()
        self.violations.clear()
        self.light_history.clear()
        self.crossed_vehicles.clear()
    
    def detect(self, frame, tracked_vehicles, frame_idx, original_frame):
        """
        Detect red light violations
        
        Args:
            frame: Frame to draw on
            tracked_vehicles: List of tracked vehicles
            frame_idx: Current frame index
            original_frame: Original frame
        
        Returns:
            List of violations found in this frame
        """
        violations_found = []
        
        # Detect traffic light state
        light_state = "GREEN"
        if self.traffic_light_box is not None:
            state = self._detect_traffic_light_state(frame, self.traffic_light_box)
            self.light_history.append(state)
            if len(self.light_history) > self.LIGHT_HISTORY_LEN:
                self.light_history.pop(0)
            light_state = max(set(self.light_history), key=self.light_history.count) if self.light_history else "GREEN"
            
            # Draw traffic light box
            x1, y1, x2, y2 = self.traffic_light_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"LIGHT:{light_state}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 0, 255) if light_state == "RED" else (0, 255, 0), 2)
        
        # Draw violation line
        if self.violation_line_y is not None:
            cv2.line(frame, (0, self.violation_line_y), 
                    (frame.shape[1], self.violation_line_y), (0, 0, 0), 3)
        
        # Check vehicles crossing on red
        if light_state == "RED" and self.violation_line_y is not None:
            for vehicle in tracked_vehicles:
                obj_id = vehicle.get('id')
                bbox = vehicle.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                
                # Check if vehicle crossed the line
                if (y1 < self.violation_line_y - 20 and 
                    y2 > self.violation_line_y and 
                    obj_id not in self.crossed_vehicles):
                    
                    self.crossed_vehicles.add(obj_id)
                    
                    if int(obj_id) not in self.seen_obj_ids_redlight:
                        snap_name = f"redlight_{obj_id}_{int(cv2.getTickCount())}.jpg"
                        snap_full_path = os.path.join(self.snapshot_folder, snap_name)
                        
                        cv2.imwrite(snap_full_path, original_frame)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Red Light Violation", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        violation_data = {
                            "type": "red_light",
                            "confidence": 0.9,
                            "bbox": [x1, y1, x2, y2],
                            "snapshot_name": snap_name,
                            "vehicle_id": int(obj_id) if obj_id is not None else None,
                            "frame": frame_idx,
                            "timestamp": frame_idx / 30.0,
                            "snapshot_url": f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                        }
                        
                        violations_found.append(violation_data)
                        self.violations.append(violation_data)
                        self.seen_obj_ids_redlight.add(int(obj_id) if obj_id is not None else 0)
        
        return violations_found
    
    def _detect_traffic_light_state(self, frame, region):
        """
        Determine traffic light state from a fixed region
        
        Args:
            frame: Full video frame (BGR)
            region: (x1, y1, x2, y2) bounding box of traffic light
        
        Returns:
            "RED" or "GREEN"
        """
        x1, y1, x2, y2 = region
        if x2 <= x1 or y2 <= y1:
            return "GREEN"
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "GREEN"
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Red masks (two ranges)
        lower_red1, upper_red1 = (0, 100, 80), (10, 255, 255)
        lower_red2, upper_red2 = (160, 100, 80), (180, 255, 255)
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Green mask
        lower_green, upper_green = (40, 50, 50), (90, 255, 255)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        r = cv2.countNonZero(mask_red)
        g = cv2.countNonZero(mask_green)
        
        # If too dark/uncertain -> assume GREEN
        if r + g < 20:
            return "GREEN"
        
        return "RED" if r > g else "GREEN"
    
    def get_violations(self):
        """Get all detected violations"""
        return self.violations.copy()
    
    def reset(self):
        """Reset detector state"""
        self.seen_obj_ids_redlight.clear()
        self.violations.clear()
        self.light_history.clear()
        self.crossed_vehicles.clear()

