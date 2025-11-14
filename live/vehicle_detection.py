import cv2
from ultralytics import YOLO
from speed_detection import calculate_speed
from helmet_triple_detection import check_helmet_triple
import numpy as np
from direction_detection import check_vehicle_direction

# ------------------- Main Detection ------------------- #
def detect_vehicles(video_source, calibration, vehicle_directions):
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

        original_frame = frame.copy()  # clean frame for cropping/helmet detection
        results_vehicle = model_vehicle.track(frame, persist=True, verbose=False)

        if results_vehicle and results_vehicle[0].boxes.id is not None:
            boxes = results_vehicle[0].boxes.xyxy.cpu().numpy()
            ids = results_vehicle[0].boxes.id.cpu().numpy()
            classes = results_vehicle[0].boxes.cls.cpu().numpy()

            # --- Speed detection (per vehicle) ---
            for box, obj_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

                speeds = calculate_speed(
                    obj_id, cx, cy, line1, line2,
                    last_positions, PIXELS_PER_METER,
                    frame_time, speeds, frame, box, original_frame
                )

                last_positions[obj_id] = (cx, cy)

                if vehicle_directions is not None:
                    if vehicle_directions==12:
                        vehicle_directions_list=[1,2]
                    if vehicle_directions==21:
                        vehicle_directions_list=[2,1]

                    check_vehicle_direction(obj_id, cy, lines, vehicle_directions_list,frame,box,original_frame)


            # --- Helmet/triple riding detection ---
            motor_boxes = []
            motor_ids = []
            rider_boxes = []

            # Draw motor and rider boxes in yellow
            for box, obj_id, cls in zip(boxes, ids, classes):
                label = model_vehicle.names[int(cls)].lower()
                x1, y1, x2, y2 = map(int, box)
                if "motor" in label:
                    motor_boxes.append(box)
                    motor_ids.append(obj_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                elif "rider" in label:
                    rider_boxes.append(box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Combine motor + overlapping rider boxes for helmet detection
            for i, motor_box in enumerate(motor_boxes):
                x1, y1, x2, y2 = map(int, motor_box)
                has_rider_overlap = False

                for rider_box in rider_boxes:
                    rx1, ry1, rx2, ry2 = map(int, rider_box)
                    if (x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1):
                        # Expand motor box to include overlapping rider
                        x1 = min(x1, rx1)
                        y1 = min(y1, ry1)
                        x2 = max(x2, rx2)
                        y2 = max(y2, ry2)
                        has_rider_overlap = True

                # Only detect helmets/triple if at least one rider overlaps
                if has_rider_overlap:
                    crop = original_frame[y1:y2, x1:x2]
                    check_helmet_triple(motor_ids[i], crop, frame,original_frame, x1, y1, x2, y2)

        # Draw calibration lines
        for p1, p2 in lines:
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Vehicle + Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
