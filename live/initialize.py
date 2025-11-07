import cv2
import math
import numpy as np
import pandas as pd
from ultralytics import YOLO

def initialize_stream(video_source="1.mp4"):
    """
    Detects vehicle motion, draws perpendicular and parallel lines,
    and performs pixel-to-meter calibration automatically using car width.
    Returns calibration data for later use.
    """
    # --- Load YOLO model ---
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 0.033

    print("[INFO] Starting motion observation for calibration...")

    paths = {}
    frame_idx = 0
    frame_limit = 80
    first_frame = None

    # -------------------- Stage 1: Detect motion direction --------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if first_frame is None:
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
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        cv2.putText(frame, f"Observing Motion... Frame {frame_idx}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Observation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_idx >= frame_limit:
            break

    cap.release()

    # -------------------- Stage 2: Compute motion + perpendiculars --------------------
    motion_lines = []
    for obj_id, points in paths.items():
        if len(points) >= 5:
            start = np.array(points[0])
            end = np.array(points[-1])
            motion_lines.append((start, end))

    if not motion_lines:
        print("[WARN] No valid vehicle motion detected.")
        return None

    output = first_frame.copy()
    perpendiculars = []

    for (start, end) in motion_lines:
        dx, dy = end - start
        mag = np.hypot(dx, dy)
        if mag < 5:
            continue
        dx /= mag
        dy /= mag

        cv2.arrowedLine(output, tuple(start.astype(int)), tuple(end.astype(int)), (255, 0, 0), 3, tipLength=0.2)

        perp_dx, perp_dy = dy, -dx
        mid = (start + end) / 2
        line_len = 100
        pt1a = (int(mid[0] - perp_dx * line_len), int(mid[1] - perp_dy * line_len))
        pt1b = (int(mid[0] + perp_dx * line_len), int(mid[1] + perp_dy * line_len))
        cv2.line(output, pt1a, pt1b, (0, 0, 255), 2)
        perpendiculars.append((mid, (perp_dx, perp_dy)))

    # -------------------- Stage 3: Select central perpendicular --------------------
    h, w, _ = output.shape
    center = np.array([w / 2, h / 2])
    distances = [np.linalg.norm(mid - center) for mid, _ in perpendiculars]
    mid_idx = int(np.argmin(distances))
    mid_pt, (perp_dx, perp_dy) = perpendiculars[mid_idx]

    line_length = max(h, w)
    p1 = (int(mid_pt[0] - perp_dx * line_length), int(mid_pt[1] - perp_dy * line_length))
    p2 = (int(mid_pt[0] + perp_dx * line_length), int(mid_pt[1] + perp_dy * line_length))
    cv2.line(output, p1, p2, (0, 255, 255), 3)

    # Draw two parallel lines
    parallel_offset = 100
    offset_vec = np.array([parallel_offset * (-perp_dy), parallel_offset * perp_dx])
    for sign in [+1, -1]:
        shift = mid_pt + sign * offset_vec
        p3 = (int(shift[0] - perp_dx * line_length), int(shift[1] - perp_dy * line_length))
        p4 = (int(shift[0] + perp_dx * line_length), int(shift[1] + perp_dy * line_length))
        cv2.line(output, p3, p4, (0, 255, 0), 2)

    # -------------------- Stage 4: Pixel-to-meter calibration --------------------
    print("[INFO] Starting pixel-to-meter calibration...")

    # Constants
    CAR_LENGTH_M = 4.5
    CLASS_ID_CAR = 2
    SAMPLE_CARS = 5
    pixel_per_meter_values = []

    # pick vertical line positions using the chosen perpendicular
    LINE_X1 = int(mid_pt[0] - offset_vec[0])
    LINE_X2 = int(mid_pt[0] + offset_vec[0])

    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.line(frame, (LINE_X1, 0), (LINE_X1, h), (0, 0, 255), 2)
        cv2.line(frame, (LINE_X2, 0), (LINE_X2, h), (0, 255, 0), 2)

        results = model.track(frame, persist=True, verbose=False)
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) == CLASS_ID_CAR:
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    if LINE_X1 < cx < LINE_X2:
                        box_w = x2 - x1
                        ppm = box_w / CAR_LENGTH_M
                        pixel_per_meter_values.append(ppm)
                        if len(pixel_per_meter_values) >= SAMPLE_CARS:
                            break
        if len(pixel_per_meter_values) >= SAMPLE_CARS:
            break

    cap.release()
    PIXELS_PER_METER = float(np.mean(pixel_per_meter_values)) if pixel_per_meter_values else 1.0
    print(f"[INFO] Calibrated scale: {PIXELS_PER_METER:.2f} pixels per meter")

    cv2.putText(output, "Motion + Perpendicular + Parallel + Calibration", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Final Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lines_pts = []
    for sign in [+1, -1]:
        shift = mid_pt + sign * offset_vec
        p3 = (int(shift[0] - perp_dx * line_length), int(shift[1] - perp_dy * line_length))
        p4 = (int(shift[0] + perp_dx * line_length), int(shift[1] + perp_dy * line_length))
        cv2.line(output, p3, p4, (0, 255, 0), 2)
        lines_pts.append((p3, p4))

    # Return calibration info
    calibration = {
        "pixels_per_meter": PIXELS_PER_METER,
        "lines": lines_pts,
        "frame_time": frame_time,
        "frame_size": (w, h)
    }

    return calibration
