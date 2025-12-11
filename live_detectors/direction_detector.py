"""
Direction Violation Detector for live detection
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from config import Config

class DirectionDetector:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
        self.road_model = YOLO(os.path.join(MODELS_DIR, 'Parking_best.pt'))
        self.vehicle_model = YOLO("yolov8n.pt")
        self.seen_obj_ids_direction = set()
        self.violations = []
        self.snapshot_folder = Config.SNAPSHOT_FOLDER
        os.makedirs(self.snapshot_folder, exist_ok=True)
        
        # Tracking state
        self.vehicle_sides = {}
        self.vehicle_sequence = {}
        self.allowed_direction = None
        self.centerline = None
        self.left_mask = None
        self.right_mask = None
    
    def initialize(self, calibration_data):
        """Initialize detector with calibration data"""
        # For direction detection, we need to detect the road centerline
        # This will be done during first frame processing
        self.seen_obj_ids_direction.clear()
        self.violations.clear()
        self.vehicle_sides.clear()
        self.vehicle_sequence.clear()
        self.allowed_direction = calibration_data.get('allowed_direction') if calibration_data else None
        self.centerline = None
        self.left_mask = None
        self.right_mask = None
        self._initialized_road = False
    
    def _initialize_road_detection(self, frame):
        """Initialize road centerline detection (called on first frame)"""
        if self._initialized_road:
            return
        
        frame_h, frame_w = frame.shape[:2]
        
        # Detect road using parking model
        results = self.road_model(frame)
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        
        for r in results:
            if hasattr(r, "masks") and r.masks is not None:
                mask = r.masks.data[0].cpu().numpy().astype(np.uint8) * 255
                mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        
        # Fill holes and process
        mask_filled = mask.copy()
        contours, hierarchy = cv2.findContours(mask_filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for j, h in enumerate(hierarchy[0]):
                cv2.drawContours(mask_filled, contours, j, 255, -1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel)
        
        bin_mask = (mask_filled > 0).astype(np.uint8)
        dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
        self.centerline = np.argmax(dist, axis=1)
        
        # Create left/right masks
        self.left_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        self.right_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        
        for y, x in enumerate(self.centerline):
            if 0 < x < frame_w:
                self.left_mask[y, :x] = 255
                self.right_mask[y, x+1:] = 255
        
        self._initialized_road = True
    
    def detect(self, frame, tracked_vehicles, frame_idx, original_frame):
        """
        Detect direction violations
        
        Args:
            frame: Frame to draw on
            tracked_vehicles: List of tracked vehicles
            frame_idx: Current frame index
            original_frame: Original frame
        
        Returns:
            List of violations found in this frame
        """
        violations_found = []
        
        # Initialize road detection on first frame
        if not self._initialized_road:
            self._initialize_road_detection(frame)
        
        if self.centerline is None:
            return violations_found
        
        frame_h, frame_w = frame.shape[:2]
        
        for vehicle in tracked_vehicles:
            obj_id = vehicle.get('id')
            bbox = vehicle.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            if cy >= len(self.centerline):
                continue
            
            cl_x = self.centerline[cy]
            
            # Initialize vehicle tracking
            if obj_id not in self.vehicle_sides:
                self.vehicle_sides[obj_id] = []
                self.vehicle_sequence[obj_id] = []
                # Determine initial side
                side = 1 if (cx - cl_x) >= 0 else -1
                self.vehicle_sides[obj_id].append(side)
                continue
            
            # Determine current side
            side = 1 if (cx - cl_x) >= 0 else -1
            last_side = self.vehicle_sides[obj_id][-1] if self.vehicle_sides[obj_id] else side
            
            # Detect crossing
            if last_side != side:
                self.vehicle_sequence[obj_id].append(1 if side > 0 else 2)
            
            self.vehicle_sides[obj_id].append(side)
            
            # Check for violation
            seq = self.vehicle_sequence[obj_id]
            violation = False
            
            if len(seq) >= 2:
                if self.allowed_direction is not None:
                    if seq[-2:] != self.allowed_direction:
                        violation = True
                else:
                    # Check if vehicle crossed wrong way
                    if (side > 0 and self.left_mask[cy, cx] > 0) or (side < 0 and self.right_mask[cy, cx] > 0):
                        violation = True
            
            if violation and int(obj_id) not in self.seen_obj_ids_direction:
                snap_name = f"direction_{obj_id}_{int(cv2.getTickCount())}.jpg"
                snap_full_path = os.path.join(self.snapshot_folder, snap_name)
                
                cv2.imwrite(snap_full_path, original_frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Direction Violation", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                violation_data = {
                    "type": "direction",
                    "confidence": 0.8,
                    "bbox": [x1, y1, x2, y2],
                    "snapshot_name": snap_name,
                    "vehicle_id": int(obj_id),
                    "frame": frame_idx,
                    "timestamp": frame_idx / 30.0,
                    "metadata": {
                        "direction": "wrong",
                        "side": "left" if side < 0 else "right"
                    },
                    "snapshot_url": f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                }
                
                violations_found.append(violation_data)
                self.violations.append(violation_data)
                self.seen_obj_ids_direction.add(int(obj_id))
        
        return violations_found
    
    def get_violations(self):
        """Get all detected violations"""
        return self.violations.copy()
    
    def reset(self):
        """Reset detector state"""
        self.seen_obj_ids_direction.clear()
        self.violations.clear()
        self.vehicle_sides.clear()
        self.vehicle_sequence.clear()
        self._initialized_road = False

