from initialize import initialize_stream
from check_direction_initialize import track_vehicle_line_order
from vehicle_detection import detect_vehicles

if __name__ == "__main__":
    # video_source = "tr.m4v"  # or RTSP URL
    video_source = "5.mp4"  # or RTSP URL
    print("[APP] Starting initialization...")
    calibration = initialize_stream(video_source)

    vehicle_data = track_vehicle_line_order(video_source, calibration)
    print("[APP] Vehicle line crossing data collected.")
    print(vehicle_data['isDirection'])
    print(vehicle_data['direction'])
  
    if calibration:
        print("[APP] Calibration complete. Starting detection system...")
        detect_vehicles(video_source, calibration, vehicle_data['direction'])
    else:
        print("[ERROR] Calibration failed.")
