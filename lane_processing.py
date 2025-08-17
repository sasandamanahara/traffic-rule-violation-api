import cv2
import json
import numpy as np
import os

# ------------------------------
# Catmull-Rom Spline
# ------------------------------
def catmull_rom_spline(P0, P1, P2, P3, n_points=20):
    t = np.linspace(0, 1, n_points).reshape(-1,1)
    t2 = t*t
    t3 = t2*t

    f1 = -0.5*t3 + t2 - 0.5*t
    f2 =  1.5*t3 - 2.5*t2 + 1.0
    f3 = -1.5*t3 + 2.0*t2 + 0.5*t
    f4 =  0.5*t3 - 0.5*t2

    points = f1*P0 + f2*P1 + f3*P2 + f4*P3
    return points

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

def process_video_lane(video_path, lane_json_path, output_dir="processed"):
    os.makedirs(output_dir, exist_ok=True)

    # Load lane JSON
    with open(lane_json_path, "r") as f:
        data = json.load(f)
    control_points = data["points"]

    # Load first frame
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("Could not read first frame from video")
    img = first_frame
    img_copy = img.copy()

    # Colors
    colors = [(0,0,255),(0,255,255),(0,255,0),(255,0,0),(255,255,0)]
    regions_pixels = []

    # Fill regions
    for i in range(len(control_points)-1):
        smooth1 = smooth_line(control_points[i])
        smooth2 = smooth_line(control_points[i+1])[::-1]

        polygon_pts = np.vstack([smooth1, smooth2]).reshape((-1,1,2))
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_pts], 255)

        color = colors[i % len(colors)]
        img_copy[mask==255] = color

        ys, xs = np.where(mask==255)
        pixels = [{"x":int(x), "y":int(y)} for x,y in zip(xs, ys)]
        regions_pixels.append(pixels)

    # Save processed image
    output_image_path = os.path.join(output_dir, "processed_frame.png")
    cv2.imwrite(output_image_path, img_copy)

    # Save pixels JSON
    pixels_json_path = os.path.join(output_dir, "smooth_regions_pixels.json")
    with open(pixels_json_path, "w") as f:
        json.dump(regions_pixels, f)

    return output_image_path, pixels_json_path
