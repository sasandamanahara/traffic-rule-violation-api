import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
import sys

# Ensure package imports resolve when loaded from different entrypoints
live_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(live_dir)
for path in (live_dir, project_root):
    if path not in sys.path:
        sys.path.insert(0, path)

# Import traffic light helper (prefer package-relative)
try:
    from live.red_light_violation.redlight_initialize import detect_traffic_light_and_line
except ImportError:
    try:
        from .red_light_violation.redlight_initialize import detect_traffic_light_and_line
    except ImportError:
        print("[WARN] Could not import detect_traffic_light_and_line")
        detect_traffic_light_and_line = None

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
        self.fps = 30.0  # Default FPS
        self.frame_size = None
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def connect(self):
        if self.cap:
            self.cap.release()

        try:
            # Create VideoCapture with appropriate backend for RTSP/RTMP
            if self.url.startswith('rtsp://') or self.url.startswith('rtmp://'):
                # Use FFMPEG backend for network streams
                print(f"[INFO] Attempting to connect to {self.url} using FFMPEG backend...")
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                
                # Set timeout for network connections
                try:
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
                except:
                    pass  # Some OpenCV versions don't support this
                
                # For RTMP, set additional properties
                if self.url.startswith('rtmp://'):
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                        # RTMP specific settings
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                    except Exception as e:
                        print(f"[WARN] Could not set RTMP properties: {e}")
            else:
                # Local file
                self.cap = cv2.VideoCapture(self.url)
            
            if not self.cap.isOpened():
                self.retry_count += 1
                if self.retry_count >= self.max_retries:
                    protocol = 'RTSP' if self.url.startswith('rtsp://') else 'RTMP' if self.url.startswith('rtmp://') else 'video source'
                    error_msg = f"[ERROR] Cannot connect to {protocol} stream: {self.url} after {self.max_retries} attempts"
                    if protocol == 'RTMP':
                        error_msg += "\n[INFO] RTMP troubleshooting:"
                        error_msg += "\n  - Ensure RTMP server is running (e.g., nginx-rtmp, OBS, etc.)"
                        error_msg += "\n  - Verify stream is actively publishing to this URL"
                        error_msg += "\n  - Check if port 1935 is accessible"
                        error_msg += "\n  - Try testing with: ffplay rtmp://localhost:1935/stream/test"
                    print(error_msg)
                    self.connected = False
                    self.connection_failed = True
                    return False
                if self.retry_count % 5 == 0:  # Log every 5th attempt
                    print(f"[WARN] Connection attempt {self.retry_count}/{self.max_retries} failed for {self.url}")
                self.connected = False
                return False
            
            # Test if we can actually read a frame (some streams open but can't read)
            test_ret, test_frame = self.cap.read()
            if not test_ret or test_frame is None:
                self.retry_count += 1
                if self.retry_count >= self.max_retries:
                    print(f"[ERROR] Stream opened but cannot read frames from {self.url}")
                    print("[INFO] This usually means the stream is not actively publishing")
                    self.cap.release()
                    self.connected = False
                    self.connection_failed = True
                    return False
                self.cap.release()
                self.connected = False
                return False
            
            # Get stream properties
            try:
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width > 0 and height > 0:
                    self.frame_size = (width, height)
            except:
                pass  # Use defaults if properties can't be read
            
            print(f"[INFO] Successfully connected to {self.url}")
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
                    # Exponential backoff for retries
                    delay = min(self.retry_delay * (1.5 ** min(self.retry_count, 5)), 30.0)
                    time.sleep(delay)
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
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

def initialize_stream(video_source):
    """
    Detects vehicle motion, draws two parallel lines for detection,
    and performs pixel-to-meter calibration automatically using car width.
    Returns calibration data for later use.
    """

    # ---------------- START STREAM ---------------- #
    stream = RTSPStream(video_source, max_retries=5)

    # Wait until stream is connected before initialization (with timeout)
    print("[APP] Waiting for stream to connect...")
    max_wait_time = 30  # 30 seconds max wait
    wait_count = 0
    while not stream.connected and not stream.connection_failed and wait_count < max_wait_time:
        time.sleep(0.5)
        wait_count += 1
    
    if stream.connection_failed or not stream.connected:
        error_reason = "connection timeout" if not stream.connection_failed else "connection failed"
        print(f"[ERROR] Failed to connect to {video_source} after {max_wait_time} seconds ({error_reason})")
        stream.stop()
        return None, None
    
    # Validate connection by reading a test frame
    test_retries = 5
    test_retry = 0
    while test_retry < test_retries:
        ret, test_frame = stream.read()
        if ret and test_frame is not None:
            break
        test_retry += 1
        time.sleep(0.5)
    
    if test_retry >= test_retries:
        print(f"[ERROR] Stream connected but cannot read frames from {video_source} (stream may not be actively publishing)")
        stream.stop()
        return None, None

    print("[INFO] Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    fps = stream.fps
    frame_time = 1 / fps if fps > 0 else 0.033

    paths = {}
    frame_idx = 0
    frame_limit = 400
    first_frame = None

    print("[INFO] Observing motion for calibration...")

    # Detect traffic light (with error handling)
    if detect_traffic_light_and_line is not None:
        try:
            traffic_light_result = detect_traffic_light_and_line(video_source)
            if traffic_light_result is not None:
                traffic_light_box = traffic_light_result.get("traffic_light_box")
                traffic_light_line = traffic_light_result.get("traffic_light_line")
            else:
                traffic_light_box = None
                traffic_light_line = None
        except Exception as e:
            print(f"[WARN] Traffic light detection failed: {str(e)}")
            traffic_light_box = None
            traffic_light_line = None
    else:
        traffic_light_box = None
        traffic_light_line = None

    while frame_idx < frame_limit:
        ret, frame = stream.read()
        if not ret or frame is None:
            # Check if connection failed
            if stream.connection_failed:
                print("[ERROR] Stream connection failed during motion observation phase of calibration")
                stream.stop()
                return None, None
            continue  # keep waiting for frames

        frame_idx += 1
        if first_frame is None:
            first_frame = frame.copy()

        try:
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
        except Exception as e:
            print(f"[WARN] Error during tracking: {str(e)}")
            continue

        # Try to display frame (may fail in headless environments)
        try:
            cv2.imshow("Motion Observation", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        except cv2.error:
            # Running in headless environment, skip display
            pass

    # Try to close windows (may fail in headless environments)
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    # Compute motion lines
    motion_lines = []
    print(f"[DEBUG] Found {len(paths)} vehicle paths")
    for obj_id, points in paths.items():
        if len(points) >= 2:
            start = np.array(points[0])
            end = np.array(points[-1])
            motion_lines.append((start, end))

    if not motion_lines:
        print("[WARN] No valid vehicle motion detected.")
        # Use default calibration values instead of failing
        print("[INFO] Using default calibration values")
        h, w, _ = first_frame.shape
        center = np.array([w / 2, h / 2])
        # Create default perpendicular line (horizontal)
        perp_dx, perp_dy = 0.0, 1.0
        mid_pt = center
        line_length = max(h, w)
        parallel_offset = 30
        offset_vec = np.array([parallel_offset * (-perp_dy), parallel_offset * perp_dx])
    else:
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

        if not perpendiculars:
            # Fallback to defaults if no valid perpendiculars
            print("[WARN] No valid perpendiculars computed, using defaults")
            h, w, _ = first_frame.shape
            center = np.array([w / 2, h / 2])
            perp_dx, perp_dy = 0.0, 1.0
            mid_pt = center
            line_length = max(h, w)
            parallel_offset = 30
            offset_vec = np.array([parallel_offset * (-perp_dy), parallel_offset * perp_dx])
        else:
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

    calibration_timeout = 0
    max_calibration_timeout = 300  # 5 minutes max for calibration

    while len(pixel_per_meter_values) < SAMPLE_CARS:
        # Check if stream connection failed
        if stream.connection_failed:
            print("[ERROR] Stream connection failed during pixel-to-meter calibration")
            stream.stop()
            return None, None
        
        ret, frame = stream.read()
        if not ret:
            calibration_timeout += 1
            if calibration_timeout > max_calibration_timeout:
                print("[WARN] Calibration timeout - using default values")
                break
            time.sleep(0.1)
            continue

        calibration_timeout = 0  # Reset timeout on successful frame read

        try:
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
        except Exception as e:
            print(f"[WARN] Error during calibration: {str(e)}")
            continue

    PIXELS_PER_METER = float(np.mean(pixel_per_meter_values)) if pixel_per_meter_values else 40.0
    if not pixel_per_meter_values:
        print("[WARN] No cars detected for calibration, using default: 40.0 pixels per meter")
    else:
        print(f"[INFO] Calibrated scale: {PIXELS_PER_METER:.2f} pixels per meter")

    # Prepare two parallel line points
    lines_pts = []
    for sign in [+1, -1]:
        shift = mid_pt + sign * offset_vec
        p1 = (int(shift[0] - perp_dx * line_length), int(shift[1] - perp_dy * line_length))
        p2 = (int(shift[0] + perp_dx * line_length), int(shift[1] + perp_dy * line_length))
        lines_pts.append((p1, p2))

    print("[INFO] Calibration completed.")
    # Don't stop the stream - it will be reused for detection
    
    # Get frame size from stream if available, otherwise use last frame
    if stream.frame_size:
        frame_size = stream.frame_size
    else:
        frame_size = (w, h)

    calibration = {
        "pixels_per_meter": PIXELS_PER_METER,
        "lines": lines_pts,
        "frame_time": frame_time,
        "frame_size": frame_size,
        "traffic_light_box": traffic_light_box,
        "traffic_light_line": traffic_light_line
    }

    # Validate calibration data before returning
    if not isinstance(calibration.get("pixels_per_meter"), (int, float)) or calibration["pixels_per_meter"] <= 0:
        calibration["pixels_per_meter"] = 40.0
    if not calibration.get("lines") or len(calibration["lines"]) != 2:
        print("[WARN] Invalid lines in calibration, using defaults")
        h, w = calibration.get("frame_size", (480, 640))
        calibration["lines"] = [
            ((0, h // 2 - 15), (w, h // 2 - 15)),
            ((0, h // 2 + 15), (w, h // 2 + 15))
        ]
    if not calibration.get("frame_size"):
        calibration["frame_size"] = (480, 640)

    return calibration, stream
