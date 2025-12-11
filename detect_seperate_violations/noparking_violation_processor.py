import cv2
import numpy as np
from ultralytics import YOLO
from flask import jsonify
import time
import os
import time
from config import Config



next_id = 0

def detect_noparking_violation_in_video(
    input_video_path,
    snapshot_folder,
    output_video_folder,
    api_base_url="/static/snapshots"
):
    seen_obj_ids_noparking = set()
    violations = []
    start_time = time.time()
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    sign_model = YOLO(os.path.join(MODELS_DIR, 'Parking_best.pt'))       # No Parking sign detection
    vehicle_model = YOLO("yolov8n.pt")           # Vehicle detection
    cap = cv2.VideoCapture(input_video_path)

    near_count = {}  # vehicle_id -> frames near the sign
    VIOLATION_FRAMES = 200
    DIST_THRESHOLD = 250

    # Simple tracker
    tracker = {}


    def assign_id(cx, cy):
        global next_id
        for vid, (px, py) in tracker.items():
            if abs(cx - px) < 40 and abs(cy - py) < 40:
                tracker[vid] = (cx, cy)
                return vid
        tracker[next_id] = (cx, cy)
        next_id += 1
        return next_id - 1


    # ---------------------------
    # Main Loop
    # ---------------------------
    frame_idx=0
    sign_results = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        if frame_idx == 1:
            sign_results = sign_model(frame, verbose=False)

        frame_idx += 1
        
        # ---------------------------
        # Detect No Parking Signs
        # ---------------------------
        
        sign_centers = []

        overlay = frame.copy()

        if sign_results is not None:
            for r in sign_results:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    # scx = (x1 + x2) // 2
                    # scy = (y1 + y2) // 2
                    sign_centers.append((x1, y2+100))

                    # Sign box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "No Parking", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # --------------------------------------
                    # Draw 50px radius mask around the sign
                    # --------------------------------------
                    cv2.circle(overlay, (x1, y2+100), DIST_THRESHOLD, (0, 0, 255), -1)



        # Blend mask (transparent red zone)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # ---------------------------
        # Detect Vehicles
        # ---------------------------
        veh_results = vehicle_model(frame, verbose=False)

        for r in veh_results:
            for box, cls_id, score in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if int(cls_id) not in [2, 3, 5, 7]:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                vehicle_id = assign_id(cx, cy)

                # Distance to nearest no-parking sign
                min_dist = 9999
                for (scx, scy) in sign_centers:
                    dist = np.sqrt((cx - scx) ** 2 + (cy - scy) ** 2)
                    min_dist = min(min_dist, dist)

                # Violation check
                if min_dist < DIST_THRESHOLD:
                    near_count[vehicle_id] = near_count.get(vehicle_id, 0) + 1
                else:
                    near_count[vehicle_id] = 0

                color = (0, 255, 0)
                if near_count.get(vehicle_id, 0) >= VIOLATION_FRAMES:
                    color = (0, 0, 255)
                    cv2.putText(frame, "PARKING VIOLATION!", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    if vehicle_id not in seen_obj_ids_noparking:
                        # Save snapshot
                        snap_path = os.path.join(snapshot_folder, f"violation_{frame_idx}_{vehicle_id}.jpg")

                        snap_name = f"violation_{frame_idx}_{vehicle_id}.jpg"

                        violations.append({
                                "id": vehicle_id,
                                "confidence": float(score),
                                "frame": frame_idx,
                                "bbox": [x1, y1, x2, y2],
                                "snapshot": snap_name,
                                "type": "Illegal Parking",
                                "snapshot_url" : f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                            })
                        
                        cv2.imwrite(snap_path, frame)
                        seen_obj_ids_noparking.add(int(vehicle_id))
                    

                
        # Skip display in headless server environment
        # if frame_idx >1:
        #     cv2.imshow("Illegal Parking Violation Detection", frame)
        #     cv2.setWindowProperty("Illegal Parking Violation Detection", cv2.WND_PROP_TOPMOST, 1)
        #     if cv2.waitKey(1) == 27:
        #         break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass


    seen_obj_ids_noparking.clear()

    return jsonify({
            'totalFrames': frame_idx,
            'processedFrames': frame_idx,
            'processingTime': round(time.time() - start_time, 2),
            'violations': violations
        })


