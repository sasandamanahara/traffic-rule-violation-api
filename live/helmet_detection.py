from ultralytics import YOLO
import cv2

model_helmet = YOLO("../models/Helmet_Detection.pt")

def check_helmet_triple(crop, frame, x1, y1):
    """
    Checks for helmet use and triple riding violations.
    """
    results_helmet = model_helmet(crop)[0]
    persons = [b for b in results_helmet.boxes.cls.cpu().numpy() if int(b) == 0]
    helmet_count = sum(1 for b in results_helmet.boxes.cls.cpu().numpy()
                       if results_helmet.names[int(b)].lower() == "helmet")

    if len(persons) > 2:
        cv2.putText(frame, "TRIPLE RIDING!", (x1, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if helmet_count < len(persons):
        cv2.putText(frame, "NO HELMET!", (x1, y1 - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
