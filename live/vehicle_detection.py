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
import queue

VIOLATION_OUTPUT_DIR = "violations/red_light"
LIGHT_HISTORY_LEN = 5
ensure_dir(VIOLATION_OUTPUT_DIR)


class RTSPStream:
    def __init__(self, url, retry_delay=2.0, max_retries=5):
        self.url = url
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.retry_count = 0
        self.cap = None
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.connected = False
        self.connection_failed = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def connect(self):
        if self.cap:
            self.cap.release()
        try:
            # Create VideoCapture with appropriate backend for RTSP/RTMP
            if self.url.startswith('rtsp://') or self.url.startswith('rtmp://'):
                # Use FFMPEG backend for network streams
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                # Set timeout for network connections
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
                # For RTMP, also set buffer size
                if self.url.startswith('rtmp://'):
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            else:
                # Local file
                self.cap = cv2.VideoCapture(self.url)
            
            if not self.cap.isOpened():
                self.retry_count += 1
                if self.retry_count >= self.max_retries:
                    protocol = 'RTSP' if self.url.startswith('rtsp://') else 'RTMP' if self.url.startswith('rtmp://') else 'video source'
                    print(f"[ERROR] Cannot connect to {protocol} stream: {self.url} after {self.max_retries} attempts")
                    self.connected = False
                    self.connection_failed = True
                    return False
                if self.retry_count % 5 == 0:  # Log every 5th attempt
                    print(f"[WARN] Connection attempt {self.retry_count}/{self.max_retries} failed for {self.url}")
                self.connected = False
                return False
            print(f"[INFO] Connected to {self.url}")
            self.connected = True
            self.retry_count = 0  # Reset on successful connection
            return True
        except Exception as e:
            self.retry_count += 1
            print(f"[EXCEPTION] during connect: {e}")
            if self.retry_count >= self.max_retries:
                self.connection_failed = True
            self.connected = False
            return False

    def update(self):
        while not self.stopped and not self.connection_failed:
            if self.cap is None or not self.cap.isOpened():
                if not self.connect():
                    if self.connection_failed:
                        break
                    time.sleep(self.retry_delay)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Only log warning occasionally to reduce spam
                if self.retry_count % 10 == 0:
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

    # Get absolute path to models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    model_path = os.path.join(models_dir, "new best.pt")
    model_vehicle = YOLO(model_path)
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

        # Try to display frame (may fail in headless environments)
        try:
            cv2.imshow("Vehicle + Speed + RLVD", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stream.stop()
                break
        except cv2.error:
            # Running in headless environment, skip display
            pass

    stream.stop()
    # Try to close windows (may fail in headless environments)
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


# ------------------- Streaming Detection (for web UI) ------------------- #
def detect_vehicles_streaming(video_source, calibration, vehicle_directions, 
                               frame_queue, violation_callback, is_running_callback, 
                               status_callback):
    """
    Streaming version of detect_vehicles that:
    - Puts frames in queue for MJPEG streaming
    - Calls violation_callback when violations are detected
    - Checks is_running_callback to know when to stop
    - Updates status via status_callback
    """
    stream = RTSPStream(video_source, max_retries=5)
    
    # Wait until stream is connected (with timeout)
    print("[APP] Waiting for stream to connect...")
    max_wait = 30  # 30 seconds max wait
    wait_count = 0
    while not stream.connected and not stream.connection_failed and wait_count < max_wait:
        time.sleep(0.5)
        wait_count += 1
    
    if stream.connection_failed or not stream.connected:
        error_msg = f"Failed to connect to video source: {video_source} after {max_wait} seconds"
        print(f"[ERROR] {error_msg}")
        status_callback(connected=False, error=error_msg)
        stream.stop()
        return
    
    status_callback(connected=True, error=None)
    
    # Get absolute path to models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    model_path = os.path.join(models_dir, "new best.pt")
    model_vehicle = YOLO(model_path)
    PIXELS_PER_METER = float(calibration.get("pixels_per_meter", 1.0))
    traffic_light_box = calibration.get("traffic_light_box", None)
    if traffic_light_box is None:
        violation_line_y = None
    else:
        violation_line_y = calibration.get("traffic_light_line", None)
    lines = calibration.get("lines", [])
    frame_time = float(calibration.get("frame_time", 0.033))
    
    if len(lines) < 2:
        print("[ERROR] Calibration must return exactly 2 parallel lines")
        status_callback(connected=False, error="Invalid calibration: missing lines")
        return
    
    line1, line2 = lines
    
    last_positions = {}
    speeds = {}
    frame_idx = 0
    light_history = []
    
    # Track violations to avoid duplicates
    speed_violations_reported = set()
    direction_violations_reported = set()
    red_light_violations_reported = set()
    helmet_violations_reported = set()
    triple_violations_reported = set()
    
    while is_running_callback():
        ret, frame = stream.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        frame_idx += 1
        original_frame = frame.copy()
        
        status_callback(frame_count=frame_idx)
        
        # --- Get traffic light state ---
        if traffic_light_box is not None:
            x1, y1, x2, y2 = traffic_light_box
            state = detect_traffic_light_state(frame, traffic_light_box)
            light_history.append(state)
            if len(light_history) > LIGHT_HISTORY_LEN:
                light_history.pop(0)
            light_state = max(set(light_history), key=light_history.count)
            
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
                
                # Speed detection
                speeds = calculate_speed(
                    obj_id, cx, cy, line1, line2,
                    last_positions, PIXELS_PER_METER,
                    frame_time, speeds, frame, box, original_frame
                )
                last_positions[obj_id] = (cx, cy)
                
                # Check for speed violation
                if obj_id in speeds and speeds[obj_id] > 30:
                    violation_key = f"speed_{obj_id}"
                    if violation_key not in speed_violations_reported:
                        speed_violations_reported.add(violation_key)
                        crop = original_frame[y1:y2, x1:x2]
                        violation_callback(
                            "speed",
                            obj_id,
                            crop,
                            bbox=[x1, y1, x2, y2],
                            metadata={"speed": speeds[obj_id]}
                        )
                
                # Direction detection
                if vehicle_directions is not None:
                    vehicle_directions_list = [1,2] if vehicle_directions==12 else [2,1]
                    is_violation = check_vehicle_direction_streaming(
                        obj_id, cy, lines, vehicle_directions_list, frame, box, original_frame
                    )
                    if is_violation:
                        violation_key = f"direction_{obj_id}"
                        if violation_key not in direction_violations_reported:
                            direction_violations_reported.add(violation_key)
                            crop = original_frame[y1:y2, x1:x2]
                            violation_callback(
                                "direction",
                                obj_id,
                                crop,
                                bbox=[x1, y1, x2, y2],
                                metadata={}
                            )
                
                tracked_objects.append({"xyxy":[x1,y1,x2,y2], "cls":int(cls), "id":obj_id, "crossed":False})
            
            # --- Helmet/triple riding detection ---
            motor_boxes, motor_ids, rider_boxes = [], [], []
            for box, obj_id, cls in zip(boxes, ids, classes):
                label = model_vehicle.names[int(cls)].lower()
                x1, y1, x2, y2 = map(int, box)
                if "motor" in label:
                    motor_boxes.append(box)
                    motor_ids.append(obj_id)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                elif "rider" in label:
                    rider_boxes.append(box)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
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
                    helmet_violation,triple_violation = check_helmet_triple(
                        motor_ids[i], crop, frame, original_frame, x1, y1, x2, y2
                    )
                    
                    if triple_violation:
                        violation_key = f"triple_{motor_ids[i]}"
                        if violation_key not in triple_violations_reported:
                            triple_violations_reported.add(violation_key)
                            violation_callback(
                                "triple_riding",
                                motor_ids[i],
                                crop,
                                bbox=[x1, y1, x2, y2],
                                metadata={}
                            )
                    
                    if helmet_violation:
                        violation_key = f"helmet_{motor_ids[i]}"
                        if violation_key not in helmet_violations_reported:
                            helmet_violations_reported.add(violation_key)
                            violation_callback(
                                "helmet",
                                motor_ids[i],
                                crop,
                                bbox=[x1, y1, x2, y2],
                                metadata={}
                            )
        
        # --- Draw violation line ---
        if violation_line_y is not None:
            draw_violation_line(frame, violation_line_y)
        
        # --- Check red light violations ---
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["xyxy"]
            y_center = (y1 + y2)//2
            if not obj["crossed"] and violation_line_y is not None and y_center >= violation_line_y and light_state=="RED":
                obj["crossed"] = True
                violation_key = f"redlight_{obj['id']}"
                if violation_key not in red_light_violations_reported:
                    red_light_violations_reported.add(violation_key)
                    crop = original_frame[y1:y2, x1:x2]
                    violation_callback(
                        "red_light",
                        obj["id"],
                        crop,
                        bbox=[x1, y1, x2, y2],
                        metadata={}
                    )
        
        # --- Draw calibration lines ---
        for p1, p2 in lines:
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,0), 2)
        
        # Put frame in queue for streaming (non-blocking)
        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            # Queue full, remove oldest and add new
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame.copy())
            except queue.Empty:
                pass
    
    stream.stop()
    status_callback(connected=False)


def check_vehicle_direction_streaming(obj_id, cy, lines, allowed_direction, frame, box, original_frame=None):
    """
    Streaming version that returns True if violation detected
    """
    from direction_detection import vehicle_sides, vehicle_sequence
    
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2
    
    # Initialize if first time
    if obj_id not in vehicle_sides:
        initial_sides = []
        for p1, p2 in lines:
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = p2[0]*p1[1] - p1[0]*p2[1]
            side = 1 if (A*cx + B*cy + C) >= 0 else -1
            initial_sides.append(side)
        
        vehicle_sides[obj_id] = initial_sides
        vehicle_sequence[obj_id] = []
        return False
    
    # Compute current sides
    new_sides = []
    for p1, p2 in lines:
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = p2[0]*p1[1] - p1[0]*p2[1]
        side = 1 if (A*cx + B*cy + C) >= 0 else -1
        new_sides.append(side)
    
    # Detect crossings
    for idx, (old, new) in enumerate(zip(vehicle_sides[obj_id], new_sides)):
        if old != new:
            vehicle_sequence[obj_id].append(idx + 1)
    
    vehicle_sides[obj_id] = new_sides
    seq = vehicle_sequence[obj_id]
    
    # Check violation
    if len(seq) >= 2:
        if allowed_direction is not None:
            if seq[-2:] != allowed_direction:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
                cv2.putText(frame, "Direction Violation", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                return True
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
            cv2.putText(frame, "Direction Violation", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return True
    
    return False


# def check_helmet_triple_streaming(obj_id, crop, frame, original_frame, x1, y1, x2, y2):
#     """
#     Streaming version that returns (helmet_violation, triple_violation) tuple
#     """
#     import helmet_triple_detection
    
#     helmet_violation = False
#     triple_violation = False
    
#     # Triple riding check
#     results_triple = helmet_triple_detection.model_triple(crop, conf=0.8)[0]
#     if len(results_triple.boxes) > 0:
#         triple_violation = True
#         cv2.putText(frame, "TRIPLE RIDING!", (x1, y1 - 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
#     # Helmet check
#     results_helmet = helmet_triple_detection.model_helmet(crop)[0]
#     helmet_count = sum(1 for b in results_helmet.boxes.cls.cpu().numpy()
#                        if results_helmet.names[int(b)].lower() == "helmet")
    
#     if helmet_count == 0:
#         helmet_violation = True
#         cv2.putText(frame, "NO HELMET!", (x1, y1 - 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
#     return helmet_violation, triple_violation
