import cv2
import math
import numpy as np
from ultralytics import YOLO

def is_inside_lines(p, line1, line2):
    """
    Returns True if point p is between two fixed parallel lines.
    line1 and line2 are tuples: ((x1,y1), (x2,y2))
    """
    p = np.array(p)
    a1, b1 = np.array(line1[0]), np.array(line1[1])
    a2 = np.array(line2[0])

    # Vector along the line
    v = b1 - a1
    v_norm = v / np.linalg.norm(v)
    # Perpendicular vector
    v_perp = np.array([-v_norm[1], v_norm[0]])
    # Distance from point to first line along perpendicular
    dist = np.dot(p - a1, v_perp)
    # Distance between lines along perpendicular
    line_dist = np.dot(a2 - a1, v_perp)

    # Check if point is between lines
    return 0 <= dist <= line_dist if line_dist > 0 else line_dist <= dist <= 0

def detect_vehicles(video_source, calibration):
    """
    Detect vehicles, estimate speed only when crossing the parallel lines,
    check helmet/triple riding, and draw calibrated reference lines.
    """
    # Load models
    model_vehicle = YOLO("../models/new best.pt")
    model_helmet = YOLO("../models/Helmet_Detection.pt")
    model_plate = YOLO("../models/Number_Plate_Detection.pt")
    model_light = YOLO("../models/yolov8n.pt")  # for traffic lights

    # Load calibration
    PIXELS_PER_METER = float(calibration.get("pixels_per_meter", 1.0))
    lines = calibration.get("lines", [])  # [(p1,p2), (p1,p2)]
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
    directions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Vehicle detection and tracking ---
        results_vehicle = model_vehicle.track(frame, persist=True, verbose=False)
        if results_vehicle and results_vehicle[0].boxes.id is not None:
            boxes = results_vehicle[0].boxes.xyxy.cpu().numpy()
            ids = results_vehicle[0].boxes.id.cpu().numpy()
            classes = results_vehicle[0].boxes.cls.cpu().numpy()

            for box, obj_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                label = model_vehicle.names[int(cls)]

                # --- Speed estimation only when inside lines ---
                if is_inside_lines((cx, cy), line1, line2):
                    if obj_id in last_positions:
                        last_cx, last_cy = last_positions[obj_id]
                        dx, dy = cx - last_cx, cy - last_cy
                        pixel_dist = math.sqrt(dx**2 + dy**2)
                        dist_m = pixel_dist / PIXELS_PER_METER
                        speed = (dist_m / frame_time) * 3.6  # km/h
                        speeds[obj_id] = 0.8 * speeds.get(obj_id, speed) + 0.2 * speed

                        # Direction violation
                        direction = "Forward" if dy > 0 else "Reverse"
                        if obj_id in directions and directions[obj_id] != direction:
                            cv2.putText(frame, "DIRECTION VIOLATION", (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        directions[obj_id] = direction
                    last_positions[obj_id] = (cx, cy)
                else:
                    # Outside lines: keep last position but don't update speed
                    last_positions[obj_id] = (cx, cy)

                # --- Helmet / Triple riding ---
                if "motor" in label.lower():
                    crop = frame[y1:y2, x1:x2]
                    results_helmet = model_helmet(crop)[0]
                    persons = [b for b in results_helmet.boxes.cls.cpu().numpy() if int(b) == 0]
                    helmet_count = sum(1 for b in results_helmet.boxes.cls.cpu().numpy()
                                       if results_helmet.names[int(b)].lower() == "helmet")

                    if len(persons) > 2:
                        cv2.putText(frame, "TRIPLE RIDING!", (x1, y1 - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if helmet_count < len(persons):
                        cv2.putText(frame, "NO HELMET!", (x1, y1 - 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # --- Draw bounding box + speed ---
                # text = f"{label} {int(obj_id)}: {speeds.get(obj_id, 0):.1f} km/h"
                text = f"{speeds.get(obj_id, 0):.1f} km/h"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- Draw reference lines ---
        for p1, p2 in lines:
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

        cv2.imshow("Vehicle + Violation Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
