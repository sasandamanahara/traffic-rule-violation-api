import cv2
import numpy as np
from ultralytics import YOLO

def initialize_stream(video_source="2.mp4"):
    """
    Tracks vehicles, observes how bounding box centers move,
    draws motion lines, perpendiculars,
    and selects the perpendicular line closest to the image center.
    Then draws two parallel lines to it.
    """
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 0.033

    print("[INFO] Starting motion observation... Move vehicles in view.")

    paths = {}
    frame_idx = 0
    frame_limit = 80
    first_frame = None

    # --- collect vehicle motion ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if first_frame is None:
            first_frame = frame.copy()

        results = model.track(frame, persist=True, verbose=False)
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, obj_id, cls in zip(boxes, ids, classes):
                if int(cls) in [2, 3, 5, 7]:  # car, motorbike, bus, truck
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    paths.setdefault(obj_id, []).append((cx, cy))

                    # draw bounding box & center
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        cv2.putText(frame, f"Observing Motion... Frame {frame_idx}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Observation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_idx >= frame_limit:
            break

    cap.release()

    # --- compute vehicle motion lines ---
    motion_lines = []
    for obj_id, points in paths.items():
        if len(points) >= 5:
            start = np.array(points[0])
            end = np.array(points[-1])
            motion_lines.append((start, end))

    if not motion_lines:
        print("[WARN] No valid vehicle motion detected.")
        return

    output = first_frame.copy()
    perpendiculars = []

    # --- draw motion and perpendicular lines ---
    for (start, end) in motion_lines:
        dx, dy = end - start
        mag = np.hypot(dx, dy)
        if mag < 5:
            continue
        dx /= mag
        dy /= mag

        # blue motion line
        cv2.arrowedLine(output, tuple(start.astype(int)), tuple(end.astype(int)),
                        (255, 0, 0), 3, tipLength=0.2)

        # perpendicular direction
        perp_dx, perp_dy = dy, -dx
        mid = (start + end) / 2
        line_len = 100

        pt1a = (int(mid[0] - perp_dx * line_len), int(mid[1] - perp_dy * line_len))
        pt1b = (int(mid[0] + perp_dx * line_len), int(mid[1] + perp_dy * line_len))
        cv2.line(output, pt1a, pt1b, (0, 0, 255), 2)

        perpendiculars.append((mid, (perp_dx, perp_dy)))

    # --- choose the perpendicular line closest to the image center ---
    h, w, _ = output.shape
    if perpendiculars:
        center = np.array([w / 2, h / 2])
        distances = [np.linalg.norm(mid - center) for mid, _ in perpendiculars]
        mid_idx = int(np.argmin(distances))
        mid_pt, (perp_dx, perp_dy) = perpendiculars[mid_idx]

        # main perpendicular line (yellow)
        line_length = max(h, w)
        p1 = (int(mid_pt[0] - perp_dx * line_length), int(mid_pt[1] - perp_dy * line_length))
        p2 = (int(mid_pt[0] + perp_dx * line_length), int(mid_pt[1] + perp_dy * line_length))
        cv2.line(output, p1, p2, (0, 255, 255), 3)

        # --- draw two parallel lines to the selected perpendicular ---
        parallel_offset = 100  # distance between lines
        # offset perpendicular to the perpendicular (i.e., along motion direction)
        offset_vec = np.array([parallel_offset * (-perp_dy), parallel_offset * perp_dx])

        for sign in [+1, -1]:
            shift = mid_pt + sign * offset_vec
            p3 = (int(shift[0] - perp_dx * line_length), int(shift[1] - perp_dy * line_length))
            p4 = (int(shift[0] + perp_dx * line_length), int(shift[1] + perp_dy * line_length))
            cv2.line(output, p3, p4, (0, 255, 0), 2)

    cv2.putText(output, "Vehicle Motion + Center Perpendicular + Parallel Lines", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Final Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output
