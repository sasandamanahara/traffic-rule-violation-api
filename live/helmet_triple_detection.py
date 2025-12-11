from ultralytics import YOLO
import cv2
import os

# Load your models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
model_helmet = YOLO(os.path.join(MODELS_DIR, 'Helmet_Detection.pt'))
model_triple = YOLO(os.path.join(MODELS_DIR, 'Triple_Riding_Detection.pt'))  # your triple riding model

def check_helmet_triple(obj_id, crop, frame, original_frame,x1, y1, x2, y2, thumb_panel, coverage_threshold=0.99):

    """
    Checks triple riding and helmet violations:
    - Triple riding: checks overlap of triple box with rider box
    - Helmet: checks how much of helmet is inside rider box
    """
    # --- Triple riding check on full frame ---
    results_triple = model_triple(original_frame, conf=0.7)[0]
    for box, conf in zip(results_triple.boxes.xyxy, results_triple.boxes.conf.cpu().numpy()):
        tx1, ty1, tx2, ty2 = map(int, box)
        triple_area = (tx2 - tx1) * (ty2 - ty1)
        if triple_area == 0:
            continue

        # Intersection with rider box
        ix1 = max(tx1, x1)
        iy1 = max(ty1, y1)
        ix2 = min(tx2, x2)
        iy2 = min(ty2, y2)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        triple_covered_ratio = inter_area / triple_area

        # Save violation if overlap insufficient
        if obj_id is not None and triple_covered_ratio >= coverage_threshold:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            crop_copy = crop.copy()
            if crop_copy.size > 0:
                crop_rgb = cv2.cvtColor(crop.copy(), cv2.COLOR_BGR2RGB)
                thumb_panel.add_thumbnail_once(obj_id, "Triple Riding Violation", crop_rgb)

    # --- Helmet check on full frame ---
    results_helmet = model_helmet(original_frame, conf=0.615)[0]
    violation_flag = False

    for box, cls, conf in zip(results_helmet.boxes.xyxy,
                               results_helmet.boxes.cls.cpu().numpy(),
                               results_helmet.boxes.conf.cpu().numpy()):
        if int(cls) != 1:
            continue  # skip non-helmet

        hx1, hy1, hx2, hy2 = map(int, box)
        helmet_area = (hx2 - hx1) * (hy2 - hy1)
        if helmet_area == 0:
            continue

        # Intersection with rider crop
        ix1 = max(hx1, x1)
        iy1 = max(hy1, y1)
        ix2 = min(hx2, x2)
        iy2 = min(hy2, y2)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        helmet_covered_ratio = inter_area / helmet_area

        if helmet_covered_ratio >= coverage_threshold:
            violation_flag = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save NO HELMET violation
    if violation_flag and obj_id is not None and helmet_covered_ratio >= coverage_threshold:
        if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop.copy(), cv2.COLOR_BGR2RGB)
                thumb_panel.add_thumbnail_once(obj_id, "Helmet Violation", crop_rgb)
