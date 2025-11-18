import cv2
from ultralytics import YOLO
from speed_detection import calculate_speed
from helmet_triple_detection import check_helmet_triple
import numpy as np
from direction_detection import check_vehicle_direction
from red_light_violation.utils import detect_traffic_light_state, draw_violation_line, ensure_dir
import threading
import time
import os

VIOLATION_OUTPUT_DIR = "violations/red_light"
LIGHT_HISTORY_LEN = 5
ensure_dir(VIOLATION_OUTPUT_DIR)


class RTSPStream:
    def __init__(self, url, retry_delay=2.0):
        self.url = url
        self.retry_delay = retry_delay
        self.cap = None
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.connected = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def connect(self):
        if self.cap:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                print(f"[ERROR] Cannot connect to {self.url}")
                self.connected = False
                return False
            print(f"[INFO] Connected to {self.url}")
            self.connected = True
            return True
        except Exception as e:
            print(f"[EXCEPTION] during connect: {e}")
            self.connected = False
            return False

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                if not self.connect():
                    time.sleep(self.retry_delay)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab frame. Reconnecting...")
                if self.cap:
                    self.cap.release()
                self.cap = None
                self.connected = False
                time.sleep(self.retry_delay)
                continue

            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        if self.cap:
            self.cap.release()


# ------------------- Main Detection ------------------- #
def detect_vehicles(video_source, calibration, vehicle_directions):

    stream = RTSPStream(video_source)

    # Wait until stream is connected before initialization
    print("[APP] Waiting for stream to connect...")
    while not stream.connected:
        time.sleep(0.5)

    model_vehicle = YOLO("../models/new best.pt")
    PIXELS_PER_METER = float(calibration.get("pixels_per_meter", 1.0))
    traffic_light_box = calibration.get("traffic_light_box", None)
    violation_line_y = calibration.get("traffic_light_line", None)
    lines = calibration.get("lines", [])
    frame_time = float(calibration.get("frame_time", 0.033))
    
    if len(lines) < 2:
        print("[ERROR] Calibration must return exactly 2 parallel lines")
        return

    line1, line2 = lines

    last_positions = {}
    speeds = {}
    frame_idx = 0
    light_history = []
    

    while True:
        ret, frame = stream.read()
        if not ret:
            break
        frame_idx += 1
        original_frame = frame.copy()  # clean frame for cropping/helmet detection

        # --- Get traffic light state from calibrated box ---
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

        # --- Detect and track vehicles ---
        results_vehicle = model_vehicle.track(frame, persist=True, verbose=False)
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

                if vehicle_directions is not None:
                    vehicle_directions_list = [1,2] if vehicle_directions==12 else [2,1]
                    check_vehicle_direction(obj_id, cy, lines, vehicle_directions_list, frame, box, original_frame)

                tracked_objects.append({"xyxy":[x1,y1,x2,y2], "cls":int(cls), "id":obj_id, "crossed":False})

            # --- Helmet/triple riding detection ---
            motor_boxes, motor_ids, rider_boxes = [], [], []
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

            for i, motor_box in enumerate(motor_boxes):
                x1, y1, x2, y2 = map(int, motor_box)
                has_rider_overlap = False
                for rider_box in rider_boxes:
                    rx1, ry1, rx2, ry2 = map(int, rider_box)
                    if (x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1):
                        x1 = min(x1, rx1); y1 = min(y1, ry1)
                        x2 = max(x2, rx2); y2 = max(y2, ry2)
                        has_rider_overlap = True
                if has_rider_overlap:
                    crop = original_frame[y1:y2, x1:x2]
                    check_helmet_triple(motor_ids[i], crop, frame, original_frame, x1, y1, x2, y2)

        # --- Draw violation line ---
        if violation_line_y is not None:
            draw_violation_line(frame, violation_line_y)

        # --- Check red light violations ---
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["xyxy"]
            y_center = (y1 + y2)//2
            if not obj["crossed"] and violation_line_y is not None and y_center >= violation_line_y and light_state=="RED":
                obj["crossed"] = True
                # Save violation info
                crop = frame[y1:y2, x1:x2]
                vid_str = "None" if obj["id"] is None else str(obj["id"])
                cv2.imwrite(os.path.join(VIOLATION_OUTPUT_DIR, f"violation_{frame_idx}_{vid_str}.jpg"), crop)

            color = (0,0,255) if obj["crossed"] else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            if obj["id"] is not None:
                cv2.putText(frame, f"ID:{obj['id']}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Draw calibration lines ---
        for p1, p2 in lines:
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,0), 2)

        # Display frame
        cv2.imshow("Vehicle + Speed + RLVD", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stream.stop()
            break

    stream.release()
    cv2.destroyAllWindows()
