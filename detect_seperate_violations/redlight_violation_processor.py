# helmet_triple_processor.py
import os
import time
import cv2
from ultralytics import YOLO
from flask import jsonify
from config import Config
import shutil
import numpy as np
# ---------------------------------------------------------------------
# GLOBAL MODELS
# ---------------------------------------------------------------------
seen_obj_ids_trafficlight = set()

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def estimate_violation_line(frame):
    """
    Detect rectangular white regions (crosswalk stripes) and choose the one
    closest to the traffic light / bottom — returns a horizontal y coordinate.
    """
    h0, w0 = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append((x,y,w,h))

    if not boxes:
        return int(h0 * 0.75)

    chosen = max(boxes, key=lambda b: b[1] + b[3])  # closest to bottom
    _, y, _, h = chosen
    violation_line = int(y + h/2)
    return violation_line

def detect_traffic_light_state(frame, region):
    """
    Determine traffic light state from a fixed region.
    
    Args:
        frame: full video frame (BGR)
        region: (x1, y1, x2, y2) bounding box of traffic light
    
    Returns:
        "RED" or "GREEN"
    """
    x1, y1, x2, y2 = region
    if x2 <= x1 or y2 <= y1:
        return "GREEN"  # invalid box

    # use slicing directly (no extra crop copy if you want even faster)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "GREEN"

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # red masks (two ranges)
    lower_red1, upper_red1 = (0, 100, 80), (10, 255, 255)
    lower_red2, upper_red2 = (160, 100, 80), (180, 255, 255)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # green mask
    lower_green, upper_green = (40, 50, 50), (90, 255, 255)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    r = cv2.countNonZero(mask_red)
    g = cv2.countNonZero(mask_green)

    # if too dark/uncertain -> assume GREEN
    if r + g < 20:
        return "GREEN"

    return "RED" if r > g else "GREEN"


def draw_violation_line(frame, y, color=(0,0,0), thickness=3):
    """
    Draw a horizontal violation line
    """
    cv2.line(frame, (0, y), (frame.shape[1], y), color, thickness)


def detect_traffic_light_and_line(video_source, num_frames=20):
    """
    Detect the traffic light box + estimate violation line from the first frame.
    Returns:
        traffic_light_box: (x1, y1, x2, y2) or None
        violation_line_y: int
    """
    print("[INFO] Detecting traffic light box & violation line...")

    # Load YOLO model (traffic light from COCO → class 9)
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_source)

    LIGHT_CLASS_ID = 9
    boxes_collected = []

    # ---- read first frame for violation line ----
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read video")
        return None, None

    # Estimate violation line once from the first frame
    violation_line_y = estimate_violation_line(first_frame)

    # ---- continue traffic light box detection ----
    count = 0
    # process the first frame as well
    frames_to_process = [first_frame]

    # read remaining frames
    while count < num_frames - 1:
        ret, frame = cap.read()
        if not ret:
            break
        frames_to_process.append(frame)
        count += 1

    # run YOLO over collected frames
    for frame in frames_to_process:
        results = model(frame, verbose=False)

        if len(results[0].boxes):
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            cls   = results[0].boxes.cls.cpu().numpy()

            for box, c in zip(xyxy, cls):
                if int(c) == LIGHT_CLASS_ID:
                    boxes_collected.append(box)

    cap.release()

    # ---- compute average traffic light box ----
    if not boxes_collected:
        print("[WARN] No traffic light detected.")
        return None, violation_line_y

    avg = np.mean(boxes_collected, axis=0)
    x1, y1, x2, y2 = map(int, avg)
    traffic_light_box = (x1, y1, x2, y2)

    return {
            "traffic_light_box": traffic_light_box,
            "traffic_light_line": violation_line_y
        }




# ---------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------
def ensure_dir(path):
    if os.path.exists(path):
        # Remove everything inside the directory
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

light_history = []
LIGHT_HISTORY_LEN = 5
# ---------------------------------------------------------------------
# MAIN PROCESSOR (returns violations list + video)
# ---------------------------------------------------------------------
def detect_redlight_violation_in_video(
    input_video_path,
    snapshot_folder,
    output_video_folder
):
    start_time = time.time()
    ensure_dir(snapshot_folder)
    ensure_dir(output_video_folder)

    traffic_light_result = detect_traffic_light_and_line(input_video_path)
    if traffic_light_result is None:
        traffic_light_box = None
        traffic_light_line = None
    else:
        traffic_light_box = traffic_light_result.get("traffic_light_box")
        traffic_light_line = traffic_light_result.get("traffic_light_line")


    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    model_vehicle = YOLO(os.path.join(MODELS_DIR, 'new best.pt'))

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

        if traffic_light_box is not None:
            x1, y1, x2, y2 = traffic_light_box
            state = detect_traffic_light_state(frame, traffic_light_box)
            light_history.append(state)
            if len(light_history) > LIGHT_HISTORY_LEN:
                light_history.pop(0)
            light_state = max(set(light_history), key=light_history.count)

            # Draw traffic light box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"LIGHT:{light_state}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,0,255) if light_state=="RED" else (0,255,0), 2)
        else:
            light_state = "GREEN"

        # vehicle tracking
        results_vehicle = model_vehicle.track(frame, persist=True, verbose=False)
        tracked_objects = []

        if results_vehicle and results_vehicle[0].boxes.id is not None:
            boxes = results_vehicle[0].boxes.xyxy.cpu().numpy()
            conf = results_vehicle[0].boxes.conf.cpu().numpy()
            ids = results_vehicle[0].boxes.id.cpu().numpy()
            classes = results_vehicle[0].boxes.cls.cpu().numpy()

            # --- Speed and direction detection ---
            for box, obj_id, cls, conf in zip(boxes, ids, classes, conf):
                x1, y1, x2, y2 = map(int, box)
                tracked_objects.append({"xyxy":[x1,y1,x2,y2], "cls":int(cls), "id":obj_id, "crossed":False, "conf":float(conf) })

        if traffic_light_line is not None:
            draw_violation_line(frame, traffic_light_line)

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["xyxy"]
            if not obj["crossed"] and traffic_light_line is not None and y1< traffic_light_line-20 and y2 > traffic_light_line and light_state=="RED":
                obj["crossed"] = True
                cv2.rectangle(frame, (x1,y1), (x2,y2),  (0,0,255), 2)
                # Save violation info
                crop = frame[y1:y2, x1:x2]
                vid_str = "None" if obj["id"] is None else str(obj["id"])
                snap_name = f"violation_{obj['id']}_{vid_str}.jpg"
                snap_full_path = os.path.join(snapshot_folder, snap_name)

                if obj['id'] not in seen_obj_ids_trafficlight:
                    cv2.imwrite(snap_full_path, frame)
                    violations.append({
                        "type": "Traffic Light",
                        "confidence": obj["conf"],
                        "bbox": [x1, y1, x2, y2],
                        "snapshot_name": snap_name,
                        "object_id": int(obj['id']) if obj['id'] is not None else None,
                        "frame": frame_idx,
                        "snapshot_url" : f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                    })
                    seen_obj_ids_trafficlight.add(obj['id'])
                
        # Skip display in headless server environment
        # cv2.imshow("Red Light Violation Detection", frame)
        # cv2.setWindowProperty("Red Light Violation Detection", cv2.WND_PROP_TOPMOST, 1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    seen_obj_ids_trafficlight.clear()
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    return jsonify({
            'totalFrames': frame_idx,
            'processedFrames': frame_idx,  # frames actually processed
            'processingTime': round(time.time() - start_time, 2),
            'violations': violations
        })

  