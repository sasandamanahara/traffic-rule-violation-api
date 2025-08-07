from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2
import numpy as np
import shutil

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

# Remove old model loading
# model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    file_id = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, file_id)
    file.save(image_path)

    # Run inference with all three models
    vehicle_results = vehicle_model(image_path)
    helmet_results = helmet_model(image_path)
    triple_results = triple_riding_model(image_path)

    # Save processed image from one of the models (e.g., vehicle)
    vehicle_results[0].save(filename=OUTPUT_IMAGE)

    # Extract data from all models
    detections = []
    for model_name, results in zip([
        'Vehicle', 'Helmet', 'Triple Riding'],
        [vehicle_results, helmet_results, triple_results]):
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            detections.append({
                'type': model_name,
                'class_id': cls_id,
                'confidence': confidence,
                'bbox': xyxy
            })

    return jsonify({
        'detections': detections,
        'image_url': f'http://localhost:5000/{OUTPUT_IMAGE}'
    })

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    file_id = f"{uuid.uuid4()}.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, file_id)
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    violations = []
    processed_frames = 0
    total_frames = 0
    start_time = cv2.getTickCount()

    # Clean up old snapshots
    shutil.rmtree(SNAPSHOT_FOLDER)
    os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vehicle_results = vehicle_model(rgb_frame)
        helmet_results = helmet_model(rgb_frame)
        triple_results = triple_riding_model(rgb_frame)
        frame_violations = []
        for model_name, results in zip([
            'Vehicle', 'Helmet', 'Triple Riding'],
            [vehicle_results, helmet_results, triple_results]):
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                frame_violations.append({
                    'type': model_name,
                    'class_id': cls_id,
                    'confidence': confidence,
                    'frame': total_frames,
                    'timestamp': round(total_frames / fps, 2),
                    'bbox': xyxy
                })
                # Draw bounding box on frame
                color = (255, 255, 0) if model_name == 'Helmet' else (0, 0, 255) if model_name == 'Triple Riding' else (255, 0, 0)
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{model_name} {(confidence*100):.1f}%"
                cv2.putText(frame, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if frame_violations:
            # Save snapshot
            snapshot_name = f"frame_{total_frames}.jpg"
            snapshot_path = os.path.join(SNAPSHOT_FOLDER, snapshot_name)
            cv2.imwrite(snapshot_path, frame)
            for v in frame_violations:
                v['snapshot_url'] = f"http://localhost:5000/{SNAPSHOT_FOLDER}/{snapshot_name}"
                violations.append(v)
        processed_frames += 1
    cap.release()
    end_time = cv2.getTickCount()
    processing_time = (end_time - start_time) / cv2.getTickFrequency()
    return jsonify({
        'totalFrames': total_frames,
        'processedFrames': processed_frames,
        'violationsDetected': len(violations),
        'processingTime': round(processing_time, 2),
        'violations': violations
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join('static', filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
