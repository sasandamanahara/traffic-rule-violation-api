from initialize import initialize_stream
from vehicle_detection import detect_vehicles

if __name__ == "__main__":
    video_source = "3.mp4"  # or RTSP URL

    print("[APP] Starting initialization...")
    calibration = initialize_stream(video_source)

    if calibration:
        print("[APP] Calibration complete. Starting detection system...")
        detect_vehicles(video_source, calibration)
    else:
        print("[ERROR] Calibration failed.")
