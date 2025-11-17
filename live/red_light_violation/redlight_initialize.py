import cv2
import numpy as np
from ultralytics import YOLO
from .line_estimator import estimate_violation_line


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
