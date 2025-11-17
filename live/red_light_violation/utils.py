import cv2, os
import numpy as np

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def detect_traffic_light_state(frame, region):
    """
    Determine traffic light state from a fixed region.
    
    Args:
        frame: full video frame (BGR)
        region: (x1, y1, x2, y2) bounding box of traffic light
    
    Returns:
        "RED" or "GREEN"
    """
    x1, y1, x2, y2 = region
    if x2 <= x1 or y2 <= y1:
        return "GREEN"  # invalid box

    # use slicing directly (no extra crop copy if you want even faster)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "GREEN"

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # red masks (two ranges)
    lower_red1, upper_red1 = (0, 100, 80), (10, 255, 255)
    lower_red2, upper_red2 = (160, 100, 80), (180, 255, 255)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # green mask
    lower_green, upper_green = (40, 50, 50), (90, 255, 255)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    r = cv2.countNonZero(mask_red)
    g = cv2.countNonZero(mask_green)

    # if too dark/uncertain -> assume GREEN
    if r + g < 20:
        return "GREEN"

    return "RED" if r > g else "GREEN"


def draw_violation_line(frame, y, color=(0,0,0), thickness=3):
    """
    Draw a horizontal violation line
    """
    cv2.line(frame, (0, y), (frame.shape[1], y), color, thickness)
