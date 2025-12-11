import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from flask import jsonify
from config import Config

seen_obj_ids_direction = set()

def detect_direction_violation_in_video(
    input_video_path,
    snapshot_folder,
    output_video_folder,
):
    start_time = time.time()

    # -----------------------
    # Ensure folders
    # -----------------------
    os.makedirs(snapshot_folder, exist_ok=True)

    # -----------------------
    # Load Models
    # -----------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    road_model = YOLO(os.path.join(MODELS_DIR, 'Parking_best.pt'))
    vehicle_model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(input_video_path)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # -----------------------
    # Pre-detect best road frame
    # -----------------------
    num_frames_to_check = 20
    morph_kernel_size = 25

    max_area = 0
    best_centerline = None

    for i in range(num_frames_to_check):
        ret, frame = cap.read()
        if not ret:
            break

        results = road_model(frame)
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        for r in results:
            if hasattr(r, "masks") and r.masks is not None:
                mask = r.masks.data[0].cpu().numpy().astype(np.uint8) * 255
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Fill holes
        mask_filled = mask.copy()
        contours, hierarchy = cv2.findContours(mask_filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for j, h in enumerate(hierarchy[0]):
                cv2.drawContours(mask_filled, contours, j, 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel)

        area = np.sum(mask_filled > 0)
        if area > max_area:
            max_area = area
            bin_mask = (mask_filled > 0).astype(np.uint8)
            dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
            best_centerline = np.argmax(dist, axis=1)

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ----------------------------------
    # Tracking setup
    # ----------------------------------
    vehicle_tracks = {}
    vehicle_first_y = {}
    vehicle_miss_count = {}
    vehicle_id_count = 0
    MAX_MISS = 5

    # Create left/right masks
    left_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    right_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

    for y, x in enumerate(best_centerline):
        if 0 < x < frame_w:
            left_mask[y, :x] = 255
            right_mask[y, x+1:] = 255

    # Violations list
    violations = []

    # ----------------------------------
    # MAIN VIDEO LOOP
    # ----------------------------------
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = vehicle_model(frame, conf=0.4)
        overlay = frame.copy()

        # Mark all tracks as "missed"
        for vid in vehicle_tracks:
            vehicle_miss_count[vid] += 1

        # --------------------------
        # Process detections
        # --------------------------
        for r in results:
            for box, cls_id, score in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                cls_id = int(cls_id)

                # Keep only vehicles
                if cls_id not in [2, 3, 5, 7]:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if cy >= len(best_centerline):
                    continue

                # Try matching to previous track
                matched_id = None
                for vid, (last_cx, last_cy) in vehicle_tracks.items():
                    if np.hypot(cx - last_cx, cy - last_cy) < 50:
                        matched_id = vid
                        break

                # New vehicle
                if matched_id is None:
                    vehicle_id_count += 1
                    matched_id = vehicle_id_count
                    vehicle_first_y[matched_id] = cy
                    vehicle_miss_count[matched_id] = 0

                # Reset miss count
                vehicle_miss_count[matched_id] = 0

                # Update tracking
                vehicle_tracks[matched_id] = (cx, cy)

                # Movement direction
                first_y = vehicle_first_y[matched_id]
                diff = cy - first_y
                direction = "none" if abs(diff) < 50 else ("up" if diff < 0 else "down")

                cl_x = best_centerline[cy]

                # Lane violation logic
                violation = False
                if direction == "down" and left_mask[cy, cx] > 0 and cx < cl_x:
                    violation = True
                elif direction == "up" and right_mask[cy, cx] > 0 and cx > cl_x:
                    violation = True

                # Determine side
                if cx < cl_x:
                    side = "left"
                else:
                    side = "right"

                # Draw and save
                if violation:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    snap_name = f"violation_{frame_idx}_{matched_id}.jpg"
                    snap_path = os.path.join(snapshot_folder, snap_name)
                    if int(matched_id) not in seen_obj_ids_direction:
                        cv2.imwrite(snap_path, overlay)

                        violations.append({
                            "id": matched_id,
                            "confidence": float(score),
                            "direction": direction,
                            "side": side,
                            "frame": frame_idx,
                            "bbox": [x1, y1, x2, y2],
                            "snapshot": snap_name,
                            "type": "Direction",
                            "snapshot_url" : f"{Config.API_BASE_URL}/static/snapshots/{snap_name}"
                        })
                        seen_obj_ids_direction.add(int(matched_id))
                # Skip display in headless server environment
                # resized = cv2.resize(overlay, None, fx=0.5, fy=0.5)
                # cv2.imshow("Direction Violation Detection", resized)
                # cv2.setWindowProperty("Direction Violation Detection", cv2.WND_PROP_TOPMOST, 1)
                # if cv2.waitKey(1) & 0xFF == 27:
                #     break

        # Remove long-missing vehicles
        to_remove = [vid for vid, miss in vehicle_miss_count.items() if miss > MAX_MISS]
        for vid in to_remove:
            del vehicle_tracks[vid]
            del vehicle_first_y[vid]
            del vehicle_miss_count[vid]


    # Cleanup
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

    seen_obj_ids_direction.clear()

    return jsonify({
            'totalFrames': frame_idx,
            'processedFrames': frame_idx,
            'processingTime': round(time.time() - start_time, 2),
            'violations': violations
        })
