import re
import cv2
import numpy as np
from ultralytics import YOLO
from flask import jsonify
import time
import os
import time
from config import Config
import easyocr
import threading
import queue
import requests

next_id = 0
reader = easyocr.Reader(['en'])
result_queue = queue.Queue()


BOT_TOKEN = "MyTockenHere"
CHAT_ID = "MyChatIDHere"
SNAPSHOT_FOLDER = Config.SNAPSHOT_FOLDER


def send_telegram_violation(violation):
    # Text message
    text = f"Violation detected!\nPlate: {violation['plate_text']}\nType: {violation['type']}\nFrame: {violation['frame']}"

    # File path
    file_path = os.path.join(SNAPSHOT_FOLDER, violation["snapshot"])
    
    # Send photo with caption
    with open(file_path, 'rb') as f:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
            data={"chat_id": CHAT_ID, "caption": text},
            files={"photo": f}
        )
    print(f"Telegram sent for {violation['plate_text']}")
    
def process_plate_async(vehicle_crop, frame_idx, vehicle_id, r_idx, p_idx, snapshot_folder,q):
    plate_crop = vehicle_crop.copy()
    plate_name = f"violation2_{frame_idx}_{vehicle_id}_plate_{r_idx}_{p_idx}.jpg"
    plate_path = os.path.join(snapshot_folder, plate_name)

    # Save and reload to get accurate OCR
    cv2.imwrite(plate_path, plate_crop)
    image = cv2.imread(plate_path)

    # Resize, grayscale, OCR
    image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, detail=0)
    text_raw = ''.join(results).upper()
    plate_text = re.sub(r'[^A-Z0-9-]', '', text_raw)


    q.put((frame_idx, plate_text))
    print(f"Detected Plate for vehicle {vehicle_id}: {plate_text}")
    


def detect_noparking_violation_in_video(
    input_video_path,
    snapshot_folder,
    output_video_folder,
    api_base_url="/static/snapshots"
):
    seen_obj_ids_noparking = set()
    violations = []
    start_time = time.time()
    sign_model = YOLO("../models/Parking_best.pt")       # No Parking sign detection
    vehicle_model = YOLO("yolov8n.pt")           # Vehicle detection
    cap = cv2.VideoCapture(input_video_path)

    near_count = {}  # vehicle_id -> frames near the sign
    VIOLATION_FRAMES = 200
    DIST_THRESHOLD = 150

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

        frame_idx += 1

        if frame_idx % 5 != 0:
            continue

        original_frame = frame.copy()   
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        if frame_idx == 5:
            sign_results = sign_model(frame, verbose=False)

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
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # cv2.putText(frame, "No Parking", (x1, y1 - 5),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # --------------------------------------
                    # Draw 50px radius mask around the sign
                    # --------------------------------------
                    # cv2.circle(overlay, (x1, y2+100), DIST_THRESHOLD, (0, 0, 255), -1)



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
                    cv2.putText(frame, "Parking Violation!", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    if vehicle_id not in seen_obj_ids_noparking:
                        # Save snapshot
                        x1_orig, y1_orig = int(x1 * 2), int(y1 * 2)
                        x2_orig, y2_orig = int(x2 * 2), int(y2 * 2)

                        # vehicle_crop = frame[y1:y2, x1:x2]
                        vehicle_crop2 = original_frame[y1_orig:y2_orig, x1_orig:x2_orig]
                        plate_model = YOLO("../models/Number_Plate_Detection.pt")
                        plate_results = plate_model(vehicle_crop2, verbose=False)


                        for r_idx, r in enumerate(plate_results):
                            for p_idx, box in enumerate(r.boxes.xyxy):
                                px1, py1, px2, py2 = map(int, box)
                                vehicle_crop2_plate = vehicle_crop2[py1:py2, px1:px2]

                                if r_idx == len(plate_results) - 1 and p_idx == len(r.boxes.xyxy) - 1:
                                    t = threading.Thread(
                                        target=process_plate_async,
                                        args=(vehicle_crop2_plate, frame_idx, vehicle_id, r_idx, p_idx, snapshot_folder, result_queue)
                                    )
                                    t.start()

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
                    

                
        if frame_idx >1:
            cv2.imshow("Illegal Parking Violation Detection", frame)
            cv2.setWindowProperty("Illegal Parking Violation Detection", cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    while not result_queue.empty():
        frame_idx_q, plate_text = result_queue.get()
        
        # Find the violation with the same frame
        for violation in violations:
            if violation["frame"] == frame_idx_q:
                violation["plate_text"] = plate_text  # add the OCR result
                send_telegram_violation(violation)
                break  # stop after finding the first match
                
    seen_obj_ids_noparking.clear()

    return jsonify({
            'totalFrames': frame_idx,
            'processedFrames': frame_idx,
            'processingTime': round(time.time() - start_time, 2),
            'violations': violations
        })


