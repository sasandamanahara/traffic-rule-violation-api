# app.py
import cv2
import threading
import time
from initialize import initialize_stream
from check_direction_initialize import track_vehicle_line_order
from vehicle_detection import detect_vehicles

RTSP_URL = "rtmp://localhost:1935/stream/test"
VIDEO_SOURCE = RTSP_URL  # can be a file like "tr.mp4" or RTSP stream
# --------------------------------------- #


# ---------------- INITIALIZATION ---------------- #
calibration = initialize_stream(VIDEO_SOURCE)

vehicle_data = track_vehicle_line_order(VIDEO_SOURCE, calibration)
print("[APP] Vehicle line crossing data collected.")
print(vehicle_data['isDirection'])
print(vehicle_data['direction'])

# ---------------- DETECTION ---------------- #
if calibration:
    print("[APP] Calibration complete. Starting detection system...")
    detect_vehicles(VIDEO_SOURCE, calibration, vehicle_data['direction'])
else:
    print("[ERROR] Calibration failed.")