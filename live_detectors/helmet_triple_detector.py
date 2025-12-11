"""
Helmet and Triple Riding Detector for live detection
"""
import os
import cv2
from ultralytics import YOLO
from config import Config

class HelmetTripleDetector:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
        self.model_helmet = YOLO(os.path.join(MODELS_DIR, 'Helmet_Detection.pt'))
        self.model_triple = YOLO(os.path.join(MODELS_DIR, 'Triple_Riding_Detection.pt'))
        self.seen_obj_ids_helmet = set()
        self.seen_obj_ids_triple = set()
        self.violations = []
        self.snapshot_folder = Config.SNAPSHOT_FOLDER
        os.makedirs(self.snapshot_folder, exist_ok=True)
    
    def initialize(self, calibration_data=None):
        """Initialize detector (no calibration needed for helmet/triple)"""
        self.seen_obj_ids_helmet.clear()
        self.seen_obj_ids_triple.clear()
        self.violations.clear()
    
    def detect(self, frame, tracked_vehicles, frame_idx, original_frame):
        """
        Detect helmet and triple riding violations
        
        Args:
            frame: Frame to draw on
            tracked_vehicles: List of tracked vehicles with motor/rider info
            frame_idx: Current frame index
            original_frame: Original frame for detection
        
        Returns:
            List of violations found in this frame
        """
        violations_found = []
        
        for vehicle in tracked_vehicles:
            if vehicle.get('type') != 'motorcycle':
                continue
            
            obj_id = vehicle.get('id')
            x1, y1, x2, y2 = vehicle.get('bbox', [0, 0, 0, 0])
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = original_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Check for violations
            frame_violations = self._check_helmet_triple(
                obj_id, crop, frame, original_frame, x1, y1, x2, y2
            )
            
            for v in frame_violations:
                v['frame'] = frame_idx
                v['timestamp'] = frame_idx / 30.0  # Approximate timestamp
                snap_full_path = os.path.join(self.snapshot_folder, v['snapshot_name'])
                
                if crop.size > 0:
                    cv2.imwrite(snap_full_path, crop)
                
                violations_found.append(v)
                self.violations.append(v)
        
        return violations_found
    
    def _check_helmet_triple(self, obj_id, crop, frame, original_frame, x1, y1, x2, y2, coverage_threshold=0.99):
        """Check for helmet and triple riding violations"""
        violations_found = []
        
        # Triple riding check
        triple_results = self.model_triple(original_frame, conf=0.783)[0]
        
        for box, conf in zip(triple_results.boxes.xyxy, triple_results.boxes.conf.cpu().numpy()):
            tx1, ty1, tx2, ty2 = map(int, box)
            triple_area = (tx2 - tx1) * (ty2 - ty1)
            if triple_area == 0:
                continue
            
            # Intersection with rider/motor merged box
            ix1 = max(tx1, x1)
            iy1 = max(ty1, y1)
            ix2 = min(tx2, x2)
            iy2 = min(ty2, y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            overlap_ratio = inter / triple_area
            
            if overlap_ratio >= coverage_threshold:
                if int(obj_id) not in self.seen_obj_ids_triple:
                    snap_name = f"triple_{obj_id}_{int(cv2.getTickCount())}.jpg"
                    violations_found.append({
                        "type": "triple_riding",
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "snapshot_name": snap_name,
                        "vehicle_id": int(obj_id),
                        "snapshot_url": f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                    })
                    self.seen_obj_ids_triple.add(int(obj_id))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Helmet check
        helmet_results = self.model_helmet(original_frame, conf=0.6)[0]
        
        for box, cls, conf in zip(
            helmet_results.boxes.xyxy,
            helmet_results.boxes.cls.cpu().numpy(),
            helmet_results.boxes.conf.cpu().numpy()
        ):
            cls = int(cls)
            if cls != 1:  # class 1 = no helmet
                continue
            
            hx1, hy1, hx2, hy2 = map(int, box)
            helmet_area = (hx2 - hx1) * (hy2 - hy1)
            if helmet_area == 0:
                continue
            
            # Intersection with merged region
            ix1 = max(hx1, x1)
            iy1 = max(hy1, y1)
            ix2 = min(hx2, x2)
            iy2 = min(hy2, y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            overlap_ratio = inter / helmet_area
            
            if overlap_ratio >= coverage_threshold:
                if int(obj_id) not in self.seen_obj_ids_helmet:
                    snap_name = f"helmet_{obj_id}_{int(cv2.getTickCount())}.jpg"
                    violations_found.append({
                        "type": "helmet",
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "snapshot_name": snap_name,
                        "vehicle_id": int(obj_id),
                        "snapshot_url": f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                    })
                    self.seen_obj_ids_helmet.add(int(obj_id))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return violations_found
    
    def get_violations(self):
        """Get all detected violations"""
        return self.violations.copy()
    
    def reset(self):
        """Reset detector state"""
        self.seen_obj_ids_helmet.clear()
        self.seen_obj_ids_triple.clear()
        self.violations.clear()

