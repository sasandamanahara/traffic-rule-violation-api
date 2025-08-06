from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_IMAGE = 'static/detected.jpg'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    file_id = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, file_id)
    file.save(image_path)

    # Run inference
    results = model(image_path)
    results[0].save(filename=OUTPUT_IMAGE)

    # Extract data
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        detections.append({'class_id': cls_id, 'confidence': confidence})

    return jsonify({
        'detections': detections,
        'image_url': f'http://localhost:5000/{OUTPUT_IMAGE}'
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join('static', filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
