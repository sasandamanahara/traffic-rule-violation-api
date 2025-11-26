from ultralytics import YOLO
import cv2
import os

# Get absolute path to models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), "models")

# Load your models with absolute paths
model_helmet = YOLO(os.path.join(models_dir, "Helmet_Detection.pt"))
model_triple = YOLO(os.path.join(models_dir, "Triple_Riding_Detection.pt"))  # your triple riding model

def check_helmet_triple(obj_id, crop, frame, original_frame, x1, y1, x2, y2, coverage_threshold=0.99):
    """
    Checks triple riding and helmet violations:
    - Triple riding: checks overlap of triple box with rider box
    - Helmet: checks how much of helmet is inside rider box
    """
    # --- Triple riding check on full frame ---
    results_triple = model_triple(original_frame, conf=0.783)[0]
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

        # Annotate based on coverage
        if triple_covered_ratio >= coverage_threshold:
            text = f"Triple Riding! conf:{conf:.6f} ratio:{triple_covered_ratio:.2f}"
            color = (0, 255, 0)
        elif inter_area > 0:
            text = f"TRIPLE RIDING! conf:{conf:.3f} ratio:{triple_covered_ratio:.2f}"
            color = (0, 0, 255)
        else:
            text = f"ERROR TRIPLE! conf:{conf:.3f} ratio:{triple_covered_ratio:.2f}"
            color = (0, 0, 255)

        # Draw on frame
        cv2.putText(frame, text, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Save violation if overlap insufficient
        if obj_id is not None and triple_covered_ratio >= coverage_threshold:
            triple_folder = os.path.join("violations", "triple_riding", f"ID_{obj_id}")
            os.makedirs(triple_folder, exist_ok=True)
            crop_copy = crop.copy()
            if crop_copy.size > 0:
                cv2.imwrite(os.path.join(triple_folder, f"triple_violation_conf{conf:.2f}.jpg"), crop_copy)
            frame_copy = original_frame.copy()
            cv2.putText(frame_copy, text, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.imwrite(os.path.join(triple_folder, f"triple_violation_full_frame_conf{conf:.2f}.jpg"), frame_copy)


    # --- Helmet check on full frame ---
    results_helmet = model_helmet(original_frame, conf=0.6)[0]
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
            text = f"No Helmet! conf:{conf:.2f} ratio:{helmet_covered_ratio:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save NO HELMET violation
    if violation_flag and obj_id is not None and helmet_covered_ratio >= coverage_threshold:
        helmet_folder = os.path.join("violations", "helmet", f"ID_{obj_id}")
        os.makedirs(helmet_folder, exist_ok=True)
        if crop.size > 0:
            cv2.imwrite(os.path.join(helmet_folder, "helmet_violation.jpg"), crop.copy())
        frame_copy = original_frame.copy()
        text = f"Helmet OK!! conf:{conf:.2f} ratio:{helmet_covered_ratio:.2f}"
        cv2.putText(frame_copy, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(helmet_folder, "helmet_violation_full_frame.jpg"), frame_copy)