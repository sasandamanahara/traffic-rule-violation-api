import platform
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math
import cv2
import threading
import time
import os
import numpy as np
import queue
from ultralytics import YOLO

# ---------- helper imports (your local modules) ----------
from speed_detection import calculate_speed
from helmet_triple_detection import check_helmet_triple
from direction_detection import check_vehicle_direction
from red_light_violation.utils import detect_traffic_light_state, draw_violation_line, ensure_dir
from thumbnail_panel import ThumbnailPanel
# --------------------------------------------------------


# constants
VIOLATION_OUTPUT_DIR = "violations/red_light"
LIGHT_HISTORY_LEN = 5
ensure_dir(VIOLATION_OUTPUT_DIR)


class VehicleDetector:
    def __init__(self, video_source, calibration, vehicle_directions):
        # Create thumbnail panel but DO NOT start its mainloop here
        self.thumb_panel = ThumbnailPanel()

        self.video_source = video_source
        self.calibration = calibration
        self.vehicle_directions = vehicle_directions

        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=5)
        self.detection_queue = queue.Queue(maxsize=5)

        # YOLO model (adjust path)
        self.model = YOLO("../models/new best.pt")
        self.stopped = False

        # Calibration
        self.PIXELS_PER_METER = float(calibration.get("pixels_per_meter", 1.0))
        self.traffic_light_box = calibration.get("traffic_light_box", None)
        self.violation_line_y = calibration.get("traffic_light_line", None)
        self.lines = calibration.get("lines", [])
        self.frame_time = float(calibration.get("frame_time", 0.033))

        if len(self.lines) < 2:
            raise ValueError("[ERROR] Calibration must return exactly 2 parallel lines")
        self.line1, self.line2 = self.lines

        # Tracking
        self.last_positions = {}
        self.speeds = {}
        self.light_history = []
        self.frame_idx = 0

    # ------------------------------
    # RTSP CONNECT WITH RETRY
    # ------------------------------
    def connect_rtsp(self):
        cap = None
        for attempt in range(1, 11):
            cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                print(f"[INFO] Connected to {self.video_source} on attempt {attempt}")
                return cap
            else:
                print(f"[WARN] Attempt {attempt}: Cannot connect. Retrying in 1s...")
                cap.release()
                time.sleep(1)
        print(f"[ERROR] Failed to connect to {self.video_source} after 10 attempts")
        return None

    # ------------------------------
    # RTSP THREAD (background)
    # ------------------------------
    def rtsp_reader(self):
        while not self.stopped:
            cap = self.connect_rtsp()
            if cap is None:
                time.sleep(1)
                continue

            while not self.stopped:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Lost connection. Reconnecting...")
                    cap.release()
                    break
                try:
                    # Put newest frame (drop if full)
                    self.frame_queue.put(frame, timeout=1)
                except queue.Full:
                    pass

            # small delay before reconnect loop
            time.sleep(0.1)

    # ------------------------------
    # YOLO INFERENCE THREAD (background)
    # ------------------------------
    def inference_thread(self):
        while not self.stopped:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # run inference (tracking) on frame
            try:
                detections = self.model.track(frame, persist=True, verbose=False)
            except Exception as e:
                print("[ERROR] model inference failed:", e)
                continue

            try:
                self.detection_queue.put((frame, detections), timeout=1)
            except queue.Full:
                pass  # drop if queue full

    # ------------------------------
    # PROCESSING LOOP (runs in MAIN thread via after)
    # ------------------------------
    def process_queues_once(self):
        """
        This method runs in the main thread (scheduled via after).
        It processes one available detection item per call (or none) and re-schedules itself.
        """
        if self.stopped:
            return

        try:
            # try to get latest available detection quickly
            frame, results_vehicle = self.detection_queue.get_nowait()
        except queue.Empty:
            # schedule next poll
            self.thumb_panel.root.after(10, self.process_queues_once)
            return

        self.frame_idx += 1
        original_frame = frame.copy()

        # --- Traffic light ---
        if self.traffic_light_box is not None:
            x1, y1, x2, y2 = self.traffic_light_box
            state = detect_traffic_light_state(frame, self.traffic_light_box)
            self.light_history.append(state)
            if len(self.light_history) > LIGHT_HISTORY_LEN:
                self.light_history.pop(0)
            light_state = max(set(self.light_history), key=self.light_history.count)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"LIGHT:{light_state}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if light_state == "RED" else (0, 255, 0), 2)
        else:
            light_state = "GREEN"

        tracked_objects = []
        motor_boxes, motor_ids, rider_boxes = [], [], []

        if results_vehicle and results_vehicle[0].boxes.id is not None:
            boxes = results_vehicle[0].boxes.xyxy.cpu().numpy()
            ids = results_vehicle[0].boxes.id.cpu().numpy()
            classes = results_vehicle[0].boxes.cls.cpu().numpy()

            for box, obj_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Speed
                self.speeds = calculate_speed(
                    obj_id, cx, cy, self.line1, self.line2,
                    self.last_positions, self.PIXELS_PER_METER,
                    self.frame_time, self.speeds, frame, box, original_frame
                )
                self.last_positions[obj_id] = (cx, cy)

                # Direction
                if self.vehicle_directions is not None:
                    vehicle_directions_list = [1, 2] if self.vehicle_directions == 12 else [2, 1]
                    check_vehicle_direction(obj_id, cy, self.lines, vehicle_directions_list,
                                            frame, box, original_frame)

                tracked_objects.append({
                    "xyxy": [x1, y1, x2, y2],
                    "cls": int(cls),
                    "id": obj_id,
                    "crossed": False
                })

                label = results_vehicle[0].names[int(cls)].lower()
                if "motor" in label:
                    motor_boxes.append(box)
                    motor_ids.append(obj_id)
                elif "rider" in label:
                    rider_boxes.append(box)

            # Helmet/triple detection
            for i, motor_box in enumerate(motor_boxes):
                x1, y1, x2, y2 = map(int, motor_box)
                has_rider_overlap = False
                for rider_box in rider_boxes:
                    rx1, ry1, rx2, ry2 = map(int, rider_box)
                    if (x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1):
                        x1 = min(x1, rx1)
                        y1 = min(y1, ry1)
                        x2 = max(x2, rx2)
                        y2 = max(y2, ry2)
                        has_rider_overlap = True
                if has_rider_overlap:
                    crop = original_frame[y1:y2, x1:x2]
                    check_helmet_triple(motor_ids[i], crop, frame, original_frame,
                                       x1, y1, x2, y2, self.thumb_panel)

        # Violation line
        if self.violation_line_y is not None and self.traffic_light_box is not None:
            draw_violation_line(frame, self.violation_line_y)

        # Red-light violation
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["xyxy"]
            y_center = (y1 + y2) // 2
            if (not obj["crossed"]
                and self.violation_line_y is not None
                and y_center >= self.violation_line_y
                and light_state == "RED"):
                obj["crossed"] = True
                crop = frame[y1:y2, x1:x2]
                vid_str = "None" if obj["id"] is None else str(obj["id"])
                save_path = os.path.join(
                    VIOLATION_OUTPUT_DIR, f"violation_{self.frame_idx}_{vid_str}.jpg"
                )
                cv2.imwrite(save_path, crop)

                # Schedule thumbnail addition safely on main thread (though we are already in main thread,
                # we use after(0, ...) to be consistent and safe)
                self.thumb_panel.root.after(0, self.thumb_panel.add_thumbnail_once,
                                            obj["id"], "redlight", save_path)

        # Calibration lines
        for p1, p2 in self.lines:
            cv2.line(frame, (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

        # Display using OpenCV (main thread)
        try:
            self.thumb_panel.update_video(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
                # close windows and exit GUI mainloop
                # cv2.destroyAllWindows()
                self.thumb_panel.root.quit()
                return
        except Exception as e:
            print("[WARN] cv2.imshow failed:", e)

        # schedule next poll quickly
        self.thumb_panel.root.after(1, self.process_queues_once)

    # ------------------------------
    # MAIN METHOD
    # ------------------------------
    def run(self):
        # Start background workers (RTSP reader + inference)
        t1 = threading.Thread(target=self.rtsp_reader, daemon=True)
        t1.start()
        t2 = threading.Thread(target=self.inference_thread, daemon=True)
        t2.start()

        # Start the processing loop in main thread via after
        self.thumb_panel.root.after(10, self.process_queues_once)

        # Finally run the Tk mainloop in the main thread (blocks here, as required)
        print("[INFO] Starting GUI mainloop (main thread).")
        self.thumb_panel.root.mainloop()
        # When mainloop exits, signal workers to stop
        self.stopped = True

        print("[INFO] GUI closed, stopping worker threads.")


# ------------------------------
# EXTERNAL ENTRY
# ------------------------------
def detect_vehicles(video_source, calibration, vehicle_directions):
    detector = VehicleDetector(video_source, calibration, vehicle_directions)
    detector.run()
