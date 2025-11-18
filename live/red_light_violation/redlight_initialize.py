import cv2
import numpy as np
from ultralytics import YOLO
from .line_estimator import estimate_violation_line
import threading
import time

# ---------------- RTSP Threaded Stream ---------------- #
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


# ---------------- TRAFFIC LIGHT & VIOLATION LINE DETECTION ---------------- #
def detect_traffic_light_and_line(video_source, num_frames=20):
    """
    Detect the traffic light box + estimate violation line from the first frame.
    Waits for stream connection if video_source is an RTSP URL.
    Returns:
        dict: {
            "traffic_light_box": (x1, y1, x2, y2) or None,
            "traffic_light_line": int
        }
    """
    print("[INFO] Detecting traffic light box & violation line...")

    # if video_source looks like RTSP, use threaded stream
    if video_source.startswith("rtmp://"):
        stream = RTSPStream(video_source)

        print("[INFO] Waiting for stream to connect...")
        while not stream.connected:
            time.sleep(0.5)

        ret, first_frame = None, None
        while ret is None or not ret:
            ret, first_frame = stream.read()
            time.sleep(0.1)

        # read additional frames if needed
        frames_to_process = [first_frame]
        count = 0
        while count < num_frames - 1:
            ret, frame = stream.read()
            if not ret:
                break
            frames_to_process.append(frame)
            count += 1

        # stream no longer needed for detection
        stream.stop()

    else:
        # local file or regular video
        cap = cv2.VideoCapture(video_source)
        ret, first_frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read video")
            return None, None

        frames_to_process = [first_frame]
        count = 0
        while count < num_frames - 1:
            ret, frame = cap.read()
            if not ret:
                break
            frames_to_process.append(frame)
            count += 1
        cap.release()

    # Estimate violation line once from the first frame
    violation_line_y = estimate_violation_line(first_frame)

    # Load YOLO model (traffic light from COCO → class 9)
    model = YOLO("yolov8n.pt")
    LIGHT_CLASS_ID = 9
    boxes_collected = []

    # run YOLO over collected frames
    for frame in frames_to_process:
        results = model(frame, verbose=False)

        if len(results[0].boxes):
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            cls   = results[0].boxes.cls.cpu().numpy()

            for box, c in zip(xyxy, cls):
                if int(c) == LIGHT_CLASS_ID:
                    boxes_collected.append(box)

    # compute average traffic light box
    if not boxes_collected:
        print("[WARN] No traffic light detected.")
        traffic_light_box = None
    else:
        avg = np.mean(boxes_collected, axis=0)
        x1, y1, x2, y2 = map(int, avg)
        traffic_light_box = (x1, y1, x2, y2)

    return {
        "traffic_light_box": traffic_light_box,
        "traffic_light_line": violation_line_y
    }
