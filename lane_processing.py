import os
import cv2
import json
import numpy as np
from ultralytics import YOLO


# ------------------------------
# Catmull-Rom Spline + Smooth Line
# ------------------------------
def catmull_rom_spline(P0, P1, P2, P3, n_points=20):
    t = np.linspace(0, 1, n_points).reshape(-1,1)
    t2 = t*t
    t3 = t2*t
    f1 = -0.5*t3 + t2 - 0.5*t
    f2 =  1.5*t3 - 2.5*t2 + 1.0
    f3 = -1.5*t3 + 2.0*t2 + 0.5*t
    f4 =  0.5*t3 - 0.5*t2
    return f1*P0 + f2*P1 + f3*P2 + f4*P3

def smooth_line(points, n_points_per_segment=20):
    pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    smooth_pts = []
    for i in range(len(pts)-1):
        P0 = pts[i-1] if i-1 >= 0 else pts[i]
        P1 = pts[i]
        P2 = pts[i+1]
        P3 = pts[i+2] if i+2 < len(pts) else pts[i+1]
        segment = catmull_rom_spline(P0, P1, P2, P3, n_points=n_points_per_segment)
        smooth_pts.extend(segment)
    return np.array(smooth_pts, dtype=np.int32)

def load_lane_regions(lane_json_path):
    with open(lane_json_path, "r") as f:
        data = json.load(f)
    control_points = data["points"]

    regions = []
    for i in range(len(control_points)-1):
        smooth1 = smooth_line(control_points[i])
        smooth2 = smooth_line(control_points[i+1])[::-1]
        polygon_pts = np.vstack([smooth1, smooth2]).reshape((-1,1,2))
        regions.append(polygon_pts)
    return regions

# ------------------------------
# Process video with YOLO and lane rules
# ------------------------------
def process_video_with_lanes(video_path, lane_json_path, output_dir="processed", yolo_model_path="models/Vehical_Detection.pt"):
    os.makedirs(output_dir, exist_ok=True)

    # Video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load lane polygons
    lane_regions = load_lane_regions(lane_json_path)

    # Output video
    output_video_path = os.path.join(output_dir, "annotated_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Load YOLO model
    model = YOLO(yolo_model_path)
    initial_positions = {}
    violations = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, conf=0.5)
        boxes = results[0].boxes
        annotated_frame = frame.copy()

        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else -1
                cls_id = int(box.cls.cpu().numpy()[0])
                x1, y1, x2, y2 = xyxy
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Check which lane
                lane_index = None
                for i, poly in enumerate(lane_regions):
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        lane_index = i
                        break
                if lane_index is None:
                    continue

                if track_id not in initial_positions:
                    initial_positions[track_id] = cy

                delta_y = cy - initial_positions[track_id]
                direction, is_violation = "", False

                # Left lanes go UP, right lanes go DOWN
                if lane_index < len(lane_regions)//2:
                    if delta_y < -50:
                        direction, is_violation = "Wrong Dir", True
                    elif delta_y > 50:
                        direction = "Correct"
                else:
                    if delta_y > 50:
                        direction, is_violation = "Wrong Dir", True
                    elif delta_y < -50:
                        direction = "Correct"

                label = f"ID:{cls_id} {direction}"
                color = (0, 0, 255) if is_violation else (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, label, (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, lineType=cv2.LINE_AA)

                if is_violation:
                    violations.append({
                        "track_id": track_id,
                        "cls_id": cls_id,
                        "lane": lane_index,
                        "cx": cx,
                        "cy": cy,
                        "direction": direction
                    })

        # Draw lanes
        for poly in lane_regions:
            cv2.polylines(annotated_frame, [poly], True, (255,255,0), 2)

        out.write(annotated_frame)

    cap.release()
    out.release()

    # Save violations
    violations_json_path = os.path.join(output_dir, "violations.json")
    with open(violations_json_path, "w") as f:
        json.dump(violations, f, indent=2)

    return output_video_path, violations_json_path

