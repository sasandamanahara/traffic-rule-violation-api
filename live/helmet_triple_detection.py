from ultralytics import YOLO
import cv2

# Load your models
model_helmet = YOLO("../models/Helmet_Detection.pt")
model_triple = YOLO("../models/Triple_Riding_Detection.pt")  # your triple riding model

def check_helmet_triple(crop, frame, x1, y1, x2, y2):
    """
    Checks for helmet use and triple riding violations using dedicated models.
    """
    # Check for triple riding
    results_triple = model_triple(crop, conf=0.8)[0]
    if len(results_triple.boxes) > 0:
        cv2.putText(frame, "TRIPLE RIDING!", (x1, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red box

    # Check for helmets
    results_helmet = model_helmet(crop)[0]
    helmet_count = sum(1 for b in results_helmet.boxes.cls.cpu().numpy()
                       if results_helmet.names[int(b)].lower() == "helmet")
    
    if helmet_count == 0:  # if no helmets detected
        cv2.putText(frame, "NO HELMET!", (x1, y1 - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
