"""
Detection and prediction API routes
"""
from flask import request, jsonify
from . import api_bp
from config import Config
import os
import uuid
import cv2
import time
from ultralytics import YOLO
import torch
import torchvision
from db_models import ViolationModel
from lane_processing import process_video_with_lanes

# Load models (these should be initialized once, not on every request)
# For now, we'll load them here, but in production, consider lazy loading
try:
    vehicle_model = YOLO("models/Vehical_Detection.pt")
    helmet_model = YOLO("models/helmet_Detection.pt")
    triple_riding_model = YOLO("models/Triple_Riding_Detection.pt")
    number_plate_model = YOLO("models/Number_Plate_Detection.pt")
    model = torch.hub.load("ultralytics/yolov5", "custom", path="models/1.pt", source="github")
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    vehicle_model = None
    helmet_model = None
    triple_riding_model = None
    number_plate_model = None
    model = None

# Helper functions
def box_belongs_to_rider(helmet_box, rider_boxes, threshold=0.9):
    """Checks if helmet_box overlaps with any rider_box by at least threshold fraction"""
    x1_h, y1_h, x2_h, y2_h = helmet_box.xyxy.cpu().numpy()[0]
    helmet_area = max(1, (x2_h - x1_h) * (y2_h - y1_h))
    max_fill_ratio = 0.0
    
    for rider_box in rider_boxes:
        x1_r, y1_r, x2_r, y2_r = rider_box.xyxy.cpu().numpy()[0]
        xi1 = max(x1_h, x1_r)
        yi1 = max(y1_h, y1_r)
        xi2 = min(x2_h, x2_r)
        yi2 = min(y2_h, y2_r)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        fill_ratio = inter_area / helmet_area
        if fill_ratio > max_fill_ratio:
            max_fill_ratio = fill_ratio
    
    return max_fill_ratio >= threshold, max_fill_ratio

def nms_boxes(boxes, iou_threshold=0.5):
    """Non-maximum suppression for boxes"""
    if not boxes:
        return []
    xyxy = []
    scores = []
    for box, fill_ratio in boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        xyxy.append([x1, y1, x2, y2])
        scores.append(float(box.conf.cpu().numpy()[0]))
    xyxy = torch.tensor(xyxy)
    scores = torch.tensor(scores)
    keep = torchvision.ops.nms(xyxy, scores, iou_threshold)
    return [boxes[i] for i in keep]

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Predict violations from an image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    file_id = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(Config.UPLOAD_FOLDER, file_id)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    file.save(image_path)

    if not vehicle_model:
        return jsonify({'error': 'Models not loaded'}), 500

    # Run inference with all four models
    vehicle_results = vehicle_model(image_path)
    helmet_results = helmet_model(image_path)
    triple_results = triple_riding_model(image_path)
    number_plate_results = number_plate_model(image_path)

    # Save processed image
    os.makedirs(os.path.dirname(Config.OUTPUT_IMAGE), exist_ok=True)
    vehicle_results[0].save(filename=Config.OUTPUT_IMAGE)

    # Extract data from all models
    detections = []
    for model_name, results in zip([
        'Vehicle', 'Helmet', 'Triple Riding', 'Number Plate'],
        [vehicle_results, helmet_results, triple_results, number_plate_results]):
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            plate_text = '' if model_name != 'Number Plate' else None
            detections.append({
                'type': model_name,
                'class_id': cls_id,
                'confidence': confidence,
                'bbox': xyxy,
                'plate_text': plate_text
            })

    return jsonify({
        'detections': detections,
        'image_url': f'{Config.API_BASE_URL}/{Config.OUTPUT_IMAGE}'
    })

@api_bp.route('/process-video', methods=['POST'])
def process_video_route():
    """Process video for violations"""
    start_time = time.time()
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    file_id = f"{uuid.uuid4()}.mp4"
    input_path = os.path.join(Config.UPLOAD_FOLDER, file_id)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = Config.OUTPUT_VIDEO_FOLDER
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"processed_{file_id}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    violation_ids = set()
    violations_data = []
    current_frame_idx = 0
    os.makedirs(Config.SNAPSHOT_FOLDER, exist_ok=True)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        current_frame_idx += 1

        # YOLOv8 models
        results_helmet = helmet_model.track(frame, persist=True, conf=0.2, tracker="botsort.yaml")
        results_tripleriding = triple_riding_model.track(frame, persist=True, conf=0.4, tracker="botsort.yaml")

        # YOLOv5 rider detection
        results_rider = model(frame)
        conf_thresh = 0.3
        rider_boxes = []
        if results_rider.xyxy[0].shape[0] > 0:
            for det in results_rider.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                if conf >= conf_thresh:
                    box = type('Box', (), {})()
                    box.xyxy = torch.tensor([[x1, y1, x2, y2]])
                    box.conf = torch.tensor([conf])
                    box.cls = torch.tensor([cls])
                    box.id = None
                    rider_boxes.append(box)

        # Collect helmet boxes with fill_ratio
        helmet_boxes = []
        for box in (results_helmet[0].boxes if results_helmet and results_helmet[0].boxes is not None else []):
            if int(box.cls.cpu().numpy()[0]) == 1:
                belongs, fill_ratio = box_belongs_to_rider(box, rider_boxes, threshold=0.9)
                if belongs:
                    helmet_boxes.append((box, fill_ratio))

        # Triple riding boxes with fill_ratio
        triple_boxes = []
        for box in (results_tripleriding[0].boxes if results_tripleriding and results_tripleriding[0].boxes is not None else []):
            belongs, fill_ratio = box_belongs_to_rider(box, rider_boxes, threshold=0.9)
            if belongs:
                triple_boxes.append((box, fill_ratio))
        
        triple_boxes = nms_boxes(triple_boxes, iou_threshold=0.5)

        # Combine all boxes
        boxes = []
        for box in helmet_boxes:
            boxes.append(('Helmet', box))
        for box in triple_boxes:
            boxes.append(('Triple Riding', box))
        for box in rider_boxes:
            boxes.append(('Rider', box))

        # Process violations and save snapshots
        for model_name, box in boxes:
            if isinstance(box, tuple):
                box_obj, fill_ratio = box
            else:
                box_obj = box
                fill_ratio = 0.0

            track_id = int(box_obj.id.cpu().numpy()[0]) if hasattr(box_obj, 'id') and box_obj.id is not None else -1
            key = (model_name, track_id)

            if model_name != 'Rider' and key not in violation_ids:
                violation_ids.add(key)

                xyxy = box_obj.xyxy.cpu().numpy().astype(int)[0].tolist()
                confidence = float(box_obj.conf.cpu().numpy()[0])
                frame_with_box = frame.copy()
                x1, y1, x2, y2 = xyxy

                color = (0, 255, 0) if model_name == 'Helmet' else (0, 255, 255) if model_name == 'Rider' else (0, 0, 255)

                cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), color, 2)
                if fill_ratio >= 0.9:
                    overlay = frame_with_box.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, frame_with_box, 0.7, 0, frame_with_box)

                label = f"{model_name} ID:{track_id} {int(fill_ratio*100)}%"
                cv2.putText(frame_with_box, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # Save snapshot
                snapshot_filename = f"{uuid.uuid4()}.jpg"
                snapshot_path = os.path.join(Config.SNAPSHOT_FOLDER, snapshot_filename)
                cv2.imwrite(snapshot_path, frame_with_box)
                snapshot_url = f"{Config.API_BASE_URL}/static/snapshots/{snapshot_filename}"

                violations_data.append({
                    'type': model_name,
                    'confidence': confidence,
                    'frame': current_frame_idx,
                    'timestamp': round(current_frame_idx / fps, 2),
                    'bbox': xyxy,
                    'snapshot_url': snapshot_url
                })

        # Draw all boxes on annotated frame
        annotated_frame = frame.copy()
        for model_name, box in boxes:
            if isinstance(box, tuple):
                box_obj, fill_ratio = box
            else:
                box_obj = box
                fill_ratio = 0.0

            x1, y1, x2, y2 = map(int, box_obj.xyxy.cpu().numpy()[0])
            color = (0, 255, 0) if model_name == 'Helmet' else (0, 255, 255) if model_name == 'Rider' else (0, 0, 255)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)

            if fill_ratio >= 0.9:
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

            track_id = int(box_obj.id.cpu().numpy()[0]) if hasattr(box_obj, 'id') and box_obj.id is not None else -1
            label = f"{model_name} ID:{track_id} {int(fill_ratio*100)}%"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(annotated_frame)

    cap.release()
    out.release()

    # Save violations to MongoDB
    saved_violation_ids = []
    if violations_data:
        mongo_violations = []
        for violation in violations_data:
            mongo_violation = {
                'type': violation['type'],
                'confidence': violation['confidence'],
                'frame': violation['frame'],
                'timestamp': violation['timestamp'],
                'bbox': violation['bbox'],
                'snapshot_url': violation['snapshot_url'],
                'video_file': file_id,
                'status': 'pending',
                'location': 'Unknown',
                'description': f"{violation['type']} detected at {violation['timestamp']}s"
            }
            mongo_violations.append(mongo_violation)
        
        saved_violation_ids = ViolationModel.create_many_violations(mongo_violations)
        if saved_violation_ids:
            print(f"✅ Saved {len(saved_violation_ids)} violations to MongoDB")
        else:
            print("⚠️  MongoDB not available, violations not saved to database")

    return jsonify({
        'totalFrames': frame_count,
        'processedFrames': frame_count,
        'violationsDetected': len(violation_ids),
        'processingTime': round(time.time() - start_time, 2),
        'violations': violations_data,
        'saved_to_db': len(saved_violation_ids) > 0,
        'db_violation_ids': saved_violation_ids
    })

@api_bp.route('/lanes', methods=['POST'])
def save_lanes():
    """Process video with lane detection"""
    if 'video' not in request.files or 'pixels_file' not in request.files:
        return jsonify({'error': 'Video or lane data file missing'}), 400

    video_file = request.files['video']
    pixels_file = request.files['pixels_file']

    # Save video
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    video_path = os.path.join(Config.UPLOAD_FOLDER, video_file.filename)
    lane_json_path = os.path.join(Config.UPLOAD_FOLDER, pixels_file.filename)
    video_file.save(video_path)
    pixels_file.save(lane_json_path)

    try:
        output_video, violations_json = process_video_with_lanes(video_path, lane_json_path)
        return jsonify({
            "message": "Processing completed",
            "output_video": output_video,
            "violations": violations_json
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

