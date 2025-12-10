# helmet_triple_processor.py
import os
import time
import cv2
from ultralytics import YOLO
from flask import jsonify
from config import Config
import shutil
import numpy as np
import math

# ---------------------------------------------------------------------
# GLOBAL MODELS
# ---------------------------------------------------------------------
# model_helmet = YOLO("../models/Helmet_Detection.pt")
# model_triple = YOLO("../models/Triple_Riding_Detection.pt")
# seen_obj_ids_helmet = set()
# seen_obj_ids_triple = set()

paths = {}
last_positions = {}


def calibration(video_source):
    frame_idx = 0
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 0.033
    model = YOLO("yolov8n.pt")
    while frame_idx < 200:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx == 1:
            first_frame = frame.copy()

        results = model.track(frame, persist=True, verbose=False)
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            for box, obj_id, cls in zip(boxes, ids, classes):
                if int(cls) in [2, 3, 5, 7]:  # car, motorbike, bus, truck
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    paths.setdefault(obj_id, []).append((cx, cy))



        # Compute motion lines
    motion_lines = []
    for obj_id, points in paths.items():
        if len(points) >= 5:
            start = np.array(points[0])
            end = np.array(points[-1])
            motion_lines.append((start, end))

    if not motion_lines:
        print("[WARN] No valid vehicle motion detected.")
        return None

    perpendiculars = []
    for (start, end) in motion_lines:
        dx, dy = end - start
        mag = np.hypot(dx, dy)
        if mag < 5:
            continue
        dx /= mag
        dy /= mag
        perp_dx, perp_dy = dy, -dx
        mid = (start + end) / 2
        perpendiculars.append((mid, (perp_dx, perp_dy)))

    # Select central perpendicular
    h, w, _ = first_frame.shape
    center = np.array([w / 2, h / 2])
    distances = [np.linalg.norm(mid - center) for mid, _ in perpendiculars]
    mid_idx = int(np.argmin(distances))
    mid_pt, (perp_dx, perp_dy) = perpendiculars[mid_idx]

    line_length = max(h, w)
    parallel_offset = 30
    offset_vec = np.array([parallel_offset * (-perp_dy), parallel_offset * perp_dx])


    # -------------------- Pixel-to-meter calibration --------------------
    print("[INFO] Starting pixel-to-meter calibration...")
    CAR_LENGTH_M = 4.5
    CLASS_ID_CAR = 2
    SAMPLE_CARS = 5
    pixel_per_meter_values = []

    # Load YOLO
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_source)

    # Define fixed vertical lines for calibration
    ret, frame = cap.read()
    if not ret:
        raise Exception("Cannot read video")

    h, w, _ = frame.shape
    LINE_X1 = int(w * 0.35)
    LINE_X2 = int(w * 0.65)

    while len(pixel_per_meter_values) < SAMPLE_CARS:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO tracking
        results = model.track(frame, persist=True, verbose=False)

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) == CLASS_ID_CAR:
                    x1, _, x2, _ = box
                    cx = (x1 + x2) / 2
                    if LINE_X1 < cx < LINE_X2:
                        box_w = x2 - x1
                        ppm = box_w / CAR_LENGTH_M
                        pixel_per_meter_values.append(ppm)
                        print(f"[DEBUG] Car detected, ppm={ppm:.2f}")
                        if len(pixel_per_meter_values) >= SAMPLE_CARS:
                            break

    cap.release()
    PIXELS_PER_METER = float(np.mean(pixel_per_meter_values)) if pixel_per_meter_values else 1.0
    print(f"[INFO] Calibrated scale: {PIXELS_PER_METER:.2f} pixels per meter")

    # Prepare two parallel line points
    lines_pts = []
    for sign in [+1, -1]:
        shift = mid_pt + sign * offset_vec
        p1 = (int(shift[0] - perp_dx * line_length), int(shift[1] - perp_dy * line_length))
        p2 = (int(shift[0] + perp_dx * line_length), int(shift[1] + perp_dy * line_length))
        lines_pts.append((p1, p2))

    print("[INFO] Calibration completed.")

    return lines_pts,PIXELS_PER_METER,frame_time

def is_inside_lines(p, line1, line2):
    """
    Returns True if point p is between two fixed parallel lines.
    line1 and line2 are tuples: ((x1,y1), (x2,y2))
    """
    p = np.array(p)
    a1, b1 = np.array(line1[0]), np.array(line1[1])
    a2 = np.array(line2[0])

    v = b1 - a1
    v_norm = v / np.linalg.norm(v)
    v_perp = np.array([-v_norm[1], v_norm[0]])
    dist = np.dot(p - a1, v_perp)
    line_dist = np.dot(a2 - a1, v_perp)

    return 0 <= dist <= line_dist if line_dist > 0 else line_dist <= dist <= 0


def calculate_speed(obj_id, cx, cy, line1, line2, last_positions, 
                    PIXELS_PER_METER, frame_time, speeds, frame, box,original_frame):
    """
    Updates speeds dictionary after calculating speed.
    Draws a blue box if speed > 30 km/h.
    Saves cropped image of speeding vehicle.
    """
    inside = is_inside_lines((cx, cy), line1, line2)
    if inside:
        if obj_id in last_positions:
            last_cx, last_cy = last_positions[obj_id]
            dx, dy = cx - last_cx, cy - last_cy
            pixel_dist = math.sqrt(dx**2 + dy**2)
            dist_m = pixel_dist / PIXELS_PER_METER
            speed = (dist_m / frame_time) * 3.6  # km/h
            speeds[obj_id] = 0.8 * speeds.get(obj_id, speed) + 0.2 * speed

            if frame is not None and box is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                current_speed = speeds[obj_id]

                # Draw bounding box and speed text
                color = (255, 0, 0) if current_speed > 30 else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{current_speed:.1f} km/h", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 0), 2)

                # --- Save speeding vehicle image ---
                if current_speed > 30:
                    folder_path = os.path.join("violations", "speed", f"ID_{obj_id}")
                    os.makedirs(folder_path, exist_ok=True)
                    crop = original_frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        filename = os.path.join(folder_path, f"speed_{int(current_speed)}.jpg")
                        cv2.imwrite(filename, crop)

    return speeds



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
def detect_speed_violation_in_video(
    input_video_path,
    snapshot_folder,
    output_video_folder,
    api_base_url="/static/snapshots"
):
    start_time = time.time()
    ensure_dir(snapshot_folder)
    ensure_dir(output_video_folder)

    model = YOLO("yolov8n.pt")
    speeds = {}
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("[ERROR] Cannot open video: " + input_video_path)

    violations = []
    frame_idx = 0

    lines,PIXELS_PER_METER, frame_time = calibration(input_video_path)
    line1, line2 = lines
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        original_frame = frame.copy()

        # vehicle tracking
        results_vehicle = model.track(frame, persist=True, verbose=False)

        tracked_objects = []

        if results_vehicle and results_vehicle[0].boxes.id is not None:
            boxes = results_vehicle[0].boxes.xyxy.cpu().numpy()
            ids = results_vehicle[0].boxes.id.cpu().numpy()
            classes = results_vehicle[0].boxes.cls.cpu().numpy()

            # --- Speed and direction detection ---
            for box, obj_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

                speeds = calculate_speed(
                    obj_id, cx, cy, line1, line2,
                    last_positions, PIXELS_PER_METER,
                    frame_time, speeds, frame, box, original_frame
                )
                last_positions[obj_id] = (cx, cy)
                
        # render window
        cv2.imshow("Vehicle + Helmet + Triple", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(violations)

    # seen_obj_ids_helmet.clear()
    # seen_obj_ids_triple.clear()

    return jsonify({
            'totalFrames': frame_idx,
            'processedFrames': frame_idx,  # frames actually processed
            'processingTime': round(time.time() - start_time, 2),
            'violations': violations
        })

    # return violations
