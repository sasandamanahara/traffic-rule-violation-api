# line_estimator.py
import cv2

def estimate_violation_line(frame):
    """
    Detect rectangular white regions (crosswalk stripes) and choose the one
    closest to the traffic light / bottom — returns a horizontal y coordinate.
    """
    h0, w0 = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append((x,y,w,h))

    if not boxes:
        return int(h0 * 0.75)

    chosen = max(boxes, key=lambda b: b[1] + b[3])  # closest to bottom
    _, y, _, h = chosen
    violation_line = int(y + h/2)
    return violation_line
