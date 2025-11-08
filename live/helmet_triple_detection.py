from ultralytics import YOLO
import cv2
import os

# Load your models
model_helmet = YOLO("../models/Helmet_Detection.pt")
model_triple = YOLO("../models/Triple_Riding_Detection.pt")  # your triple riding model

def check_helmet_triple(obj_id, crop, frame,original_frame, x1, y1, x2, y2):
    """
    Checks for helmet use and triple riding violations using dedicated models.
    Draws warnings on `frame` but saves the full frame with only the violation box highlighted.
    """
    # --- Triple riding check ---
    results_triple = model_triple(crop, conf=0.8)[0]
    if len(results_triple.boxes) > 0:
        # Draw on visualization frame
        cv2.putText(frame, "TRIPLE RIDING!", (x1, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Save clean crop in triple riding folder
        if obj_id is not None:
            triple_folder = os.path.join("violations", "triple_riding", f"ID_{obj_id}")
            os.makedirs(triple_folder, exist_ok=True)
            clean_crop = crop.copy()
            if clean_crop.size > 0:
                filename_crop = os.path.join(triple_folder, "triple_violation.jpg")
                cv2.imwrite(filename_crop, clean_crop)

            # --- Save full frame with only this bounding box ---
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            filename_frame = os.path.join(triple_folder, "triple_violation_full_frame.jpg")
            cv2.imwrite(filename_frame, frame_copy)

    # --- Helmet check ---
    results_helmet = model_helmet(crop)[0]
    helmet_count = sum(1 for b in results_helmet.boxes.cls.cpu().numpy()
                       if results_helmet.names[int(b)].lower() == "helmet")
    
    if helmet_count == 0:  # no helmets detected
        # Draw on visualization frame
        cv2.putText(frame, "NO HELMET!", (x1, y1 - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Save clean crop in helmet violation folder
        if obj_id is not None:
            helmet_folder = os.path.join("violations", "helmet", f"ID_{obj_id}")
            os.makedirs(helmet_folder, exist_ok=True)
            clean_crop = crop.copy()
            if clean_crop.size > 0:
                filename_crop = os.path.join(helmet_folder, "helmet_violation.jpg")
                cv2.imwrite(filename_crop, clean_crop)

            # --- Save full frame with only this bounding box ---
            frame_copy = original_frame.copy()
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            filename_frame = os.path.join(helmet_folder, "helmet_violation_full_frame.jpg")
            cv2.imwrite(filename_frame, frame_copy)
