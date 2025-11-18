import cv2
import numpy as np
from ultralytics import YOLO
from red_light_violation.redlight_initialize import detect_traffic_light_and_line
import threading
import time

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

def initialize_stream(video_source):
    """
    Detects vehicle motion, draws two parallel lines for detection,
    and performs pixel-to-meter calibration automatically using car width.
    Returns calibration data for later use.
    """

    # ---------------- START STREAM ---------------- #
    stream = RTSPStream(video_source)

    # Wait until stream is connected before initialization
    print("[APP] Waiting for stream to connect...")
    while not stream.connected:
        time.sleep(0.5)


    print("[INFO] Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 0.033

    paths = {}
    frame_idx = 0
    frame_limit = 400
    first_frame = None

    print("[INFO] Observing motion for calibration...")

    traffic_light_result = detect_traffic_light_and_line(video_source)

    traffic_light_box = traffic_light_result["traffic_light_box"]
    traffic_light_line = traffic_light_result["traffic_light_line"]


    while frame_idx < frame_limit:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue  # keep waiting for frames

        frame_idx += 1
        if first_frame is None:
            first_frame = frame.copy()

        results = model.track(frame, persist=True, verbose=False)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, obj_id, cls in zip(boxes, ids, classes):
                if int(cls) in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    paths.setdefault(obj_id, []).append((cx, cy))

        cv2.imshow("Motion Observation", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


    # Compute motion lines
    motion_lines = []
    print(len(paths))
    for obj_id, points in paths.items():
        if len(points) >= 2:
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

    h, w, _ = frame.shape
    LINE_X1 = int(w * 0.35)
    LINE_X2 = int(w * 0.65)

    while len(pixel_per_meter_values) < SAMPLE_CARS:
        ret, frame = stream.read()
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
    stream.stop()
    calibration = {
        "pixels_per_meter": PIXELS_PER_METER,
        "lines": lines_pts,
        "frame_time": frame_time,
        "frame_size": (w, h),
        "traffic_light_box": traffic_light_box,
        "traffic_light_line": traffic_light_line
    }

    return calibration
