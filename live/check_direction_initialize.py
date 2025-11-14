
import cv2
import numpy as np
from ultralytics import YOLO

def track_vehicle_line_order(video_source, calibration, frame_limit=300):
    """
    Tracks vehicles and records the sequence of line crossings.
    Crossing is detected based purely on side change of the vehicle center
    relative to the line, no distance threshold needed.
    Returns vehicle_data and isDirection.
    """
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_source)
    lines = calibration["lines"]  # list of line points [(p1, p2), ...]

    vehicle_data = {}  # obj_id: {"crossed_seq": [], "positions": [], "last_side": []}
    frame_idx = 0

    while frame_idx < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.track(frame, persist=True, verbose=False)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, obj_id, cls in zip(boxes, ids, classes):
                if int(cls) not in [2, 3, 5, 7]:  # car, motorbike, bus, truck
                    continue

                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2)/2, (y1 + y2)/2  # vehicle center

                # Initialize vehicle data if first detection
                if obj_id not in vehicle_data:
                    last_side = []
                    for p1, p2 in lines:
                        A = p2[1] - p1[1]
                        B = p1[0] - p2[0]
                        C = p2[0]*p1[1] - p1[0]*p2[1]
                        side = 1 if (A*cx + B*cy + C) >= 0 else -1
                        last_side.append(side)

                    vehicle_data[obj_id] = {
                        "crossed_seq": [],
                        "positions": [(cx, cy)],
                        "last_side": last_side
                    }
                else:
                    vehicle_data[obj_id]["positions"].append((cx, cy))

                # --- Check line crossings based on side change ---
                for idx, (p1, p2) in enumerate(lines):
                    A = p2[1] - p1[1]
                    B = p1[0] - p2[0]
                    C = p2[0]*p1[1] - p1[0]*p2[1]
                    side = 1 if (A*cx + B*cy + C) >= 0 else -1

                    if side != vehicle_data[obj_id]["last_side"][idx]:
                        vehicle_data[obj_id]["crossed_seq"].append(idx + 1)  # 1-based
                        vehicle_data[obj_id]["last_side"][idx] = side

                # --- Draw bounding box and center ---
                color = (0, 255, 255)  # yellow
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                label = f"ID:{obj_id} seq: {vehicle_data[obj_id]['crossed_seq']}"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Draw line(s) ---
        for i, (p1, p2) in enumerate(lines):
            line_color = (0, 0, 255)  # red
            cv2.line(frame, p1, p2, line_color, 2)

        cv2.imshow("Vehicle Line Crossing", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- Determine direction summary ---
    direction12 = []
    direction21 = []

    for obj_id, data in vehicle_data.items():
        seq = data["crossed_seq"]

        # Check only if at least two crossings
        if len(seq) >= 2:
            if seq[-2:] == [1, 2] and obj_id not in direction12:
                direction12.append(obj_id)
            elif seq[-2:] == [2, 1] and obj_id not in direction21:
                direction21.append(obj_id)

    # --- Final direction logic ---
    if len(direction12) == 0 and len(direction21) > 0:
        isDirection = True
        direction = 21

    elif len(direction21) == 0 and len(direction12) > 0:
        isDirection = True
        direction = 12

    else:
        isDirection = False
        direction = None

    # print("direction12:", direction12)
    # print("direction21:", direction21)
    # print("isDirection:", isDirection)
    # print("direction:", direction)

    return {
        "vehicle_data": vehicle_data,
        "direction12": direction12,
        "direction21": direction21,
        "isDirection": isDirection,
        "direction": direction
    }

