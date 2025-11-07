import math
import cv2
import numpy as np


def is_inside_lines(p, line1, line2):
    """
    Returns True if point p is between two fixed parallel lines.
    line1 and line2 are tuples: ((x1,y1), (x2,y2))
    """
    p = np.array(p)
    a1, b1 = np.array(line1[0]), np.array(line1[1])
    a2 = np.array(line2[0])

    v = b1 - a1
    v_norm = v / np.linalg.norm(v)
    v_perp = np.array([-v_norm[1], v_norm[0]])
    dist = np.dot(p - a1, v_perp)
    line_dist = np.dot(a2 - a1, v_perp)

    return 0 <= dist <= line_dist if line_dist > 0 else line_dist <= dist <= 0


def calculate_speed(obj_id, cx, cy,line1,line2, last_positions, PIXELS_PER_METER, frame_time, speeds, frame, box):
    """
    Updates speeds dictionary after calculating speed.
    Draws a blue box if speed > 30 km/h.
    """
    inside = is_inside_lines((cx, cy), line1, line2)
    if inside:
        if obj_id in last_positions:
            last_cx, last_cy = last_positions[obj_id]
            dx, dy = cx - last_cx, cy - last_cy
            pixel_dist = math.sqrt(dx**2 + dy**2)
            dist_m = pixel_dist / PIXELS_PER_METER
            speed = (dist_m / frame_time) * 3.6  # km/h
            speeds[obj_id] = 0.8 * speeds.get(obj_id, speed) + 0.2 * speed

            if frame is not None and box is not None:
                x1, y1, x2, y2 = [int(v) for v in box]  # ensure integers
                if speeds[obj_id] > 30:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue box
                # Always show speed text
                cv2.putText(frame, f"{speeds[obj_id]:.1f} km/h", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return speeds
