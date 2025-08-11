from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2
import numpy as np
import shutil
import time

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_IMAGE = 'static/detected.jpg'
SNAPSHOT_FOLDER = 'static/snapshots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Load models
vehicle_model = YOLO("models/Vehical_Detection.pt")
helmet_model = YOLO("models/helmet_Detection.pt")
triple_riding_model = YOLO("models/Triple_Riding_Detection.pt")
number_plate_model = YOLO("models/Number_Plate_Detection.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    file_id = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, file_id)
    file.save(image_path)

    # Run inference with all four models
    vehicle_results = vehicle_model(image_path)
    helmet_results = helmet_model(image_path)
    triple_results = triple_riding_model(image_path)
    number_plate_results = number_plate_model(image_path)

    # Save processed image from one of the models (e.g., vehicle)
    vehicle_results[0].save(filename=OUTPUT_IMAGE)

    # Extract data from all models
    detections = []
    for model_name, results in zip([
        'Vehicle', 'Helmet', 'Triple Riding', 'Number Plate'],
        [vehicle_results, helmet_results, triple_results, number_plate_results]):
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            plate_text = '' if model_name != 'Number Plate' else None  # Placeholder for OCR
            detections.append({
                'type': model_name,
                'class_id': cls_id,
                'confidence': confidence,
                'bbox': xyxy,
                'plate_text': plate_text
            })

    return jsonify({
        'detections': detections,
        'image_url': f'http://localhost:5000/{OUTPUT_IMAGE}'
    })

@app.route('/process-video', methods=['POST'])
def process_video():
    start_time = time.time()
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    file_id = f"{uuid.uuid4()}.mp4"
    input_path = os.path.join(UPLOAD_FOLDER, file_id)
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = 'output_videos'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"processed_{file_id}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results_helmet = helmet_model.track(frame, persist=True, conf=0.1)
        results_tripleriding = triple_riding_model.track(frame, persist=True, conf=0.7)

        boxes = []

        helmet_boxes = [
            box for box in (results_helmet[0].boxes if results_helmet and results_helmet[0].boxes is not None else [])
            if int(box.cls.cpu().numpy()[0]) == 1
        ]
        triple_boxes = results_tripleriding[0].boxes if results_tripleriding and results_tripleriding[0].boxes is not None else []

        for box in helmet_boxes:
            boxes.append(('Helmet', box))

        for box in triple_boxes:
            boxes.append(('Triple Riding', box))

        violation_ids = set()
        
        for model_name, box in boxes:
            track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else -1
            violation_ids.add((model_name, track_id))

        violations_data = []
        for model_name, box in boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0].tolist()   # [x1, y1, x2, y2]
            confidence = float(box.conf.cpu().numpy()[0])           # float
            track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else -1

            # Copy the frame so original is not changed
            frame_with_box = frame.copy()

            # Draw bounding box on the copied frame
            x1, y1, x2, y2 = xyxy
            color = (0, 255, 0) if model_name == 'Helmet' else (0, 0, 255)
            cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), color, thickness=2)

            # Save snapshot image
            snapshot_filename = f"{uuid.uuid4()}.jpg"
            snapshot_path = os.path.join(SNAPSHOT_FOLDER, snapshot_filename)
            cv2.imwrite(snapshot_path, frame_with_box)

            # Public URL for the snapshot
            snapshot_url = f"http://localhost:5000/static/snapshots/{snapshot_filename}"

            violations_data.append({
                'type': model_name,
                'confidence': confidence,
                'frame': frame_count,
                'timestamp': round(frame_count / fps, 2),
                'bbox': xyxy,
                'plate_text': "plate_text",
                'snapshot_url': snapshot_url
            })


            
        annotated_frame = frame.copy()

        
        for model_name, box in boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            track_id = int(box.id.cpu().numpy()[0]) if box.id is not None else -1
            confidence = float(box.conf.cpu().numpy()[0])  # confidence score

            x1, y1, x2, y2 = xyxy
            # label = f"{model_name} {confidence*100:.1f}%"

            # Choose color per model or violation status
            color = (0, 255, 0) if model_name == 'Helmet' else (0, 0, 255)  # green for helmet, red for triple riding

            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness=1)
            # cv2.putText(
            #     annotated_frame, cls_id, (x1, y2 + 10),
            #     cv2.FONT_HERSHEY_SIMPLEX, 1 , color, 2, lineType=cv2.LINE_AA
            # )


        # Write the annotated frame to output
        out.write(annotated_frame)

    cap.release()
    out.release()

    return jsonify({
        'totalFrames': frame_count,
        'processedFrames': frame_count,
        'violationsDetected': len(violation_ids),
        'processingTime': round(time.time() - start_time, 2),
        'violations': violations_data
    })


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join('static', filename), mimetype='image/jpeg')


@app.route('/output_videos/<path:filename>')
def serve_processed_video(filename):
    return send_file(os.path.join('output_videos', filename), mimetype='video/mp4')


if __name__ == '__main__':
    app.run(debug=False)

