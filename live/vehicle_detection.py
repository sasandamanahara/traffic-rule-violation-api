import cv2
import numpy as np
from ultralytics import YOLO
from speed_detection import calculate_speed
from helmet_detection import check_helmet_triple

# ------------------- Utility ------------------- #
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

def draw_bounding_box(frame, x1, y1, x2, y2, speed, show_speed=True):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if show_speed:
        cv2.putText(frame, f"{speed:.1f} km/h", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# ------------------- Main Detection ------------------- #
def detect_vehicles(video_source, calibration):
    model_vehicle = YOLO("../models/new best.pt")
    PIXELS_PER_METER = float(calibration.get("pixels_per_meter", 1.0))
    lines = calibration.get("lines", [])
    frame_time = float(calibration.get("frame_time", 0.033))
    if len(lines) < 2:
        print("[ERROR] Calibration must return exactly 2 parallel lines")
        return

    line1, line2 = lines
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video:", video_source)
        return

    last_positions = {}
    speeds = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results_vehicle = model_vehicle.track(frame, persist=True, verbose=False)
        if results_vehicle and results_vehicle[0].boxes.id is not None:
            boxes = results_vehicle[0].boxes.xyxy.cpu().numpy()
            ids = results_vehicle[0].boxes.id.cpu().numpy()
            classes = results_vehicle[0].boxes.cls.cpu().numpy()

            for box, obj_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                label = model_vehicle.names[int(cls)]

                # --- Only calculate speed if center is inside the lines ---
                inside = is_inside_lines((cx, cy), line1, line2)
                if inside:
                    speeds = calculate_speed(obj_id, cx, cy, last_positions, PIXELS_PER_METER, frame_time, speeds)

                # Update last positions anyway for tracking continuity
                last_positions[obj_id] = (cx, cy)

                # Helmet/triple riding detection for motorcycles
                if "motor" in label.lower():
                    crop = frame[y1:y2, x1:x2]
                    check_helmet_triple(crop, frame, x1, y1)

                # Draw bounding box and speed only if inside
                draw_bounding_box(frame, x1, y1, x2, y2, speeds.get(obj_id, 0), show_speed=inside)

        # Draw calibration lines
        for p1, p2 in lines:
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

        cv2.imshow("Vehicle + Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
