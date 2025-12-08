# helmet_triple_processor.py
import os
import time
import cv2
from ultralytics import YOLO
from flask import jsonify
from config import Config
import shutil
# ---------------------------------------------------------------------
# GLOBAL MODELS
# ---------------------------------------------------------------------
model_helmet = YOLO("../models/Helmet_Detection.pt")
model_triple = YOLO("../models/Triple_Riding_Detection.pt")
seen_obj_ids_helmet = set()
seen_obj_ids_triple = set()

# ---------------------------------------------------------------------
# HELMET + TRIPLE CHECK
# ---------------------------------------------------------------------
def check_helmet_triple(obj_id, crop, original_frame, x1, y1, x2, y2, coverage_threshold=0.99):
    violations_found = []

    # ================================================================
    # TRIPLE RIDING CHECK
    # ================================================================
    triple_results = model_triple(original_frame, conf=0.783)[0]

    for box, conf in zip(triple_results.boxes.xyxy,
                         triple_results.boxes.conf.cpu().numpy()):

        tx1, ty1, tx2, ty2 = map(int, box)

        triple_area = (tx2 - tx1) * (ty2 - ty1)
        if triple_area == 0:
            continue

        # intersection with rider/motor merged box
        ix1 = max(tx1, x1)
        iy1 = max(ty1, y1)
        ix2 = min(tx2, x2)
        iy2 = min(ty2, y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        overlap_ratio = inter / triple_area

        if overlap_ratio >= coverage_threshold:

            if int(obj_id) not in seen_obj_ids_triple:
                snap_name = f"triple_{obj_id}.jpg"

                violations_found.append({
                    "type": "Triple Riding",
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                    "snapshot_name": snap_name,
                    "object_id": int(obj_id),
                    "snapshot_url" : f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                })
                seen_obj_ids_triple.add(int(obj_id))


    # ================================================================
    # HELMET CHECK
    # ================================================================
    helmet_results = model_helmet(original_frame, conf=0.6)[0]

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

        # intersection with merged region
        ix1 = max(hx1, x1)
        iy1 = max(hy1, y1)
        ix2 = min(hx2, x2)
        iy2 = min(hy2, y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        overlap_ratio = inter / helmet_area
        
        if overlap_ratio >= coverage_threshold:

            # Before the loop, keep a set of seen object IDs
            

            # Inside your loop over YOLO detections:
            if int(obj_id) not in seen_obj_ids_helmet:
                snap_name = f"helmet_{obj_id}.jpg"
                violations_found.append({
                    "type": "Helmet",
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                    "snapshot_name": snap_name,
                    "object_id": int(obj_id),
                    "snapshot_url" : f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                })
                seen_obj_ids_helmet.add(int(obj_id))

    return violations_found



# ---------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------
def ensure_dir(path):
    if os.path.exists(path):
        # Remove everything inside the directory
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)



# ---------------------------------------------------------------------
# MAIN PROCESSOR (returns violations list + video)
# ---------------------------------------------------------------------
def detect_violations_in_video(
    input_video_path,
    snapshot_folder,
    output_video_folder,
    api_base_url="/static/snapshots"
):
    start_time = time.time()
    ensure_dir(snapshot_folder)
    ensure_dir(output_video_folder)

    model_vehicle = YOLO("../models/new best.pt")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("[ERROR] Cannot open video: " + input_video_path)

    violations = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        original_frame = frame.copy()

        # vehicle tracking
        results_vehicle = model_vehicle.track(frame, persist=True, verbose=False)

        if results_vehicle and results_vehicle[0].boxes.id is not None:

            boxes = results_vehicle[0].boxes.xyxy.cpu().numpy()
            ids = results_vehicle[0].boxes.id.cpu().numpy()
            classes = results_vehicle[0].boxes.cls.cpu().numpy()

            motor_boxes, motor_ids, rider_boxes = [], [], []

            for box, obj_id, cls in zip(boxes, ids, classes):
                label = model_vehicle.names[int(cls)].lower()
                x1, y1, x2, y2 = map(int, box)

                if "motor" in label:
                    motor_boxes.append(box)
                    motor_ids.append(obj_id)

                elif "rider" in label:
                    rider_boxes.append(box)

            # MERGE motor + rider and check helmet/triple
            for i, motor_box in enumerate(motor_boxes):
                x1, y1, x2, y2 = map(int, motor_box)
                merged = False

                for rider_box in rider_boxes:
                    rx1, ry1, rx2, ry2 = map(int, rider_box)

                    if (x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1):
                        x1 = min(x1, rx1)
                        y1 = min(y1, ry1)
                        x2 = max(x2, rx2)
                        y2 = max(y2, ry2)
                        merged = True

                if merged:
                    crop = original_frame[y1:y2, x1:x2]

                    violations_found_in_frame = check_helmet_triple(
                        motor_ids[i], crop, original_frame, x1, y1, x2, y2
                    )

                    # --- save snapshots + append violations ---
                    for v in violations_found_in_frame:
                        snap_full_path = os.path.join(snapshot_folder, v["snapshot_name"])

                        if crop.size > 0:
                            cv2.imwrite(snap_full_path, crop)

                        v["frame"] = frame_idx

                        violations.append(v)

        # render window
        cv2.imshow("Vehicle + Helmet + Triple", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(violations)

    seen_obj_ids_helmet.clear()
    seen_obj_ids_triple.clear()

    return jsonify({
            'totalFrames': frame_idx,
            'processedFrames': frame_idx,  # frames actually processed
            'processingTime': round(time.time() - start_time, 2),
            'violations': violations
        })

    # return violations
