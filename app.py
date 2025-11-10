from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2
import numpy as np
import shutil
import time
import torch
import torchvision 
import jwt
import bcrypt
from functools import wraps
from datetime import datetime, timedelta
from lane_processing import process_video_with_lanes
from database import get_db, get_database_info
from db_models import ViolationModel, CameraModel, AdminModel

app = Flask(__name__)

# Configure CORS to allow requests from frontend
# In development, allow all origins (change in production)
CORS(app, 
     origins="*",
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     expose_headers=["Content-Type", "Authorization"],
     supports_credentials=False)

# Add after_request handler to ensure CORS headers are always set
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS,PATCH')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# JWT Secret Key (in production, use environment variable)
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

# Initialize database connection
db = get_db()

# ========================================
# Authentication Helper Functions
# ========================================

def hash_password(password):
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    """Verify a password against a hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_token(admin_id, username):
    """Generate JWT token for admin"""
    payload = {
        'admin_id': admin_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm=JWT_ALGORITHM)

def verify_token(token):
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # Check for token in Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid authorization header format'
                }), 401
        
        if not token:
            return jsonify({
                'success': False,
                'error': 'Authentication token is missing'
            }), 401
        
        payload = verify_token(token)
        if not payload:
            return jsonify({
                'success': False,
                'error': 'Invalid or expired token'
            }), 401
        
        # Add admin info to request context
        request.current_admin = payload
        return f(*args, **kwargs)
    
    return decorated_function

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

# Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
model = torch.hub.load("ultralytics/yolov5", "custom", path="models/1.pt", source="github")


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
        'image_url': f'http://localhost:5001/{OUTPUT_IMAGE}'
    })

def box_belongs_to_rider(helmet_box, rider_boxes, threshold=0.9):
    """
    Checks if helmet_box overlaps with any rider_box by at least `threshold` fraction of the helmet area.
    Returns:
        belongs (bool)  : True if overlap >= threshold
        fill_ratio (float): Fraction of helmet area overlapped (0.0 to 1.0)
    """
    # Get helmet coordinates
    x1_h, y1_h, x2_h, y2_h = helmet_box.xyxy.cpu().numpy()[0]
    helmet_area = max(1, (x2_h - x1_h) * (y2_h - y1_h))  # prevent division by zero

    max_fill_ratio = 0.0

    for rider_box in rider_boxes:
        # Get rider coordinates
        x1_r, y1_r, x2_r, y2_r = rider_box.xyxy.cpu().numpy()[0]
        
        # Intersection coordinates
        xi1 = max(x1_h, x1_r)
        yi1 = max(y1_h, y1_r)
        xi2 = min(x2_h, x2_r)
        yi2 = min(y2_h, y2_r)

        # Intersection area
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Fill ratio: fraction of helmet area that overlaps rider box
        fill_ratio = inter_area / helmet_area

        # Track maximum ratio among all riders
        if fill_ratio > max_fill_ratio:
            max_fill_ratio = fill_ratio

    # Return both boolean and numeric ratio
    return max_fill_ratio >= threshold, max_fill_ratio

def nms_boxes(boxes, iou_threshold=0.5):
    if not boxes:
        return []
    # Convert to xyxy + confidence
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

def count_riders_in_box(triple_box, rider_boxes):
    x1_t, y1_t, x2_t, y2_t = triple_box.xyxy.cpu().numpy()[0]
    count = 0
    for rider in rider_boxes:
        x1_r, y1_r, x2_r, y2_r = rider.xyxy.cpu().numpy()[0]
        # check if rider center is inside triple box
        cx, cy = (x1_r + x2_r) / 2, (y1_r + y2_r) / 2
        if x1_t <= cx <= x2_t and y1_t <= cy <= y2_t:
            count += 1
    return count

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

    violation_ids = set()
    violations_data = []
    current_frame_idx = 0

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

        # triple_boxes = [
        #     box for box in triple_boxes
        #     if count_riders_in_box(box[0], rider_boxes) >= 3
        # ]

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
            # Unpack tuple if needed
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

                # Draw rectangle + highlight if fill >= 90%
                cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), color, 2)
                if fill_ratio >= 0.9:
                    overlay = frame_with_box.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, frame_with_box, 0.7, 0, frame_with_box)

                # Draw label + fill %
                label = f"{model_name} ID:{track_id} {int(fill_ratio*100)}%"
                cv2.putText(frame_with_box, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # Save snapshot
                snapshot_filename = f"{uuid.uuid4()}.jpg"
                snapshot_path = os.path.join(SNAPSHOT_FOLDER, snapshot_filename)
                cv2.imwrite(snapshot_path, frame_with_box)
                snapshot_url = f"http://localhost:5001/static/snapshots/{snapshot_filename}"

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

            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)

            # Highlight if ≥90%
            if fill_ratio >= 0.9:
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

            # Draw label + fill %
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
        # Prepare violations for MongoDB
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
                'location': 'Unknown',  # Update with actual location if available
                'description': f"{violation['type']} detected at {violation['timestamp']}s"
            }
            mongo_violations.append(mongo_violation)
        
        # Save to MongoDB
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




@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join('static', filename), mimetype='image/jpeg')


@app.route('/output_videos/<path:filename>')
def serve_processed_video(filename):
    return send_file(os.path.join('output_videos', filename), mimetype='video/mp4')

@app.route('/api/lanes', methods=['POST'])
def save_lanes():
    if 'video' not in request.files or 'pixels_file' not in request.files:
        return jsonify({'error': 'Video or lane data file missing'}), 400

    video_file = request.files['video']
    pixels_file = request.files['pixels_file']

    # Save video
    os.makedirs("uploads", exist_ok=True)
    video_path = os.path.join("uploads", video_file.filename)
    lane_json_path = os.path.join("uploads", pixels_file.filename)
    video_file.save(video_path)

    # Save lane data JSON file
    lane_data_path = os.path.join("uploads", pixels_file.filename)
    pixels_file.save(lane_data_path)

    try:
        output_video, violations_json = process_video_with_lanes(video_path, lane_json_path)
        return jsonify({
            "message": "Processing completed",
            "output_video": output_video,
            "violations": violations_json
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/processed/<path:filename>')
def serve_video(filename):
    return send_file(os.path.join('processed', filename), mimetype='video/mp4', as_attachment=False)


# ========================================
# Authentication API Endpoints
# ========================================

# Handle OPTIONS preflight requests
@app.route('/api/auth/register', methods=['OPTIONS'])
def register_options():
    return '', 200

@app.route('/api/auth/register', methods=['POST'])
def register_admin():
    """Register a new admin user"""
    try:
        data = request.json
        
        # Validate required fields
        if not data or not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({
                'success': False,
                'error': 'Username, email, and password are required'
            }), 400
        
        # Check if admin already exists
        existing = AdminModel.get_admin_by_username(data['username'])
        if existing:
            return jsonify({
                'success': False,
                'error': 'Username already exists'
            }), 400
        
        existing = AdminModel.get_admin_by_email(data['email'])
        if existing:
            return jsonify({
                'success': False,
                'error': 'Email already exists'
            }), 400
        
        # Hash password
        hashed_password = hash_password(data['password'])
        
        # Create admin
        admin_data = {
            'username': data['username'],
            'email': data['email'],
            'password': hashed_password,
            'role': data.get('role', 'admin'),
            'is_active': True
        }
        
        admin_id = AdminModel.create_admin(admin_data)
        
        if admin_id:
            # Generate token
            token = generate_token(admin_id, data['username'])
            
            return jsonify({
                'success': True,
                'message': 'Admin registered successfully',
                'token': token,
                'admin': {
                    'id': admin_id,
                    'username': data['username'],
                    'email': data['email'],
                    'role': admin_data['role']
                }
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create admin'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/login', methods=['POST'])
def login_admin():
    """Login admin user"""
    try:
        data = request.json
        
        # Validate required fields
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({
                'success': False,
                'error': 'Username and password are required'
            }), 400
        
        # Get admin by username
        admin = AdminModel.get_admin_by_username(data['username'])
        
        if not admin:
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        # Check if admin is active
        if not admin.get('is_active', True):
            return jsonify({
                'success': False,
                'error': 'Account is deactivated'
            }), 403
        
        # Verify password
        if not verify_password(data['password'], admin['password']):
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        # Generate token
        token = generate_token(admin['_id'], admin['username'])
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'token': token,
            'admin': {
                'id': admin['_id'],
                'username': admin['username'],
                'email': admin.get('email', ''),
                'role': admin.get('role', 'admin')
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/verify', methods=['POST'])
def verify_auth():
    """Verify authentication token"""
    try:
        data = request.json
        token = data.get('token') if data else None
        
        # Also check Authorization header
        if not token and 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]
            except IndexError:
                pass
        
        if not token:
            return jsonify({
                'success': False,
                'error': 'Token is required'
            }), 400
        
        payload = verify_token(token)
        
        if not payload:
            return jsonify({
                'success': False,
                'error': 'Invalid or expired token'
            }), 401
        
        # Get admin info
        admin = AdminModel.get_admin_by_id(payload['admin_id'])
        
        if not admin or not admin.get('is_active', True):
            return jsonify({
                'success': False,
                'error': 'Admin not found or inactive'
            }), 401
        
        return jsonify({
            'success': True,
            'valid': True,
            'admin': {
                'id': admin['_id'],
                'username': admin['username'],
                'email': admin.get('email', ''),
                'role': admin.get('role', 'admin')
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_current_admin():
    """Get current authenticated admin info"""
    try:
        admin_id = request.current_admin['admin_id']
        admin = AdminModel.get_admin_by_id(admin_id)
        
        if not admin:
            return jsonify({
                'success': False,
                'error': 'Admin not found'
            }), 404
        
        return jsonify({
            'success': True,
            'admin': {
                'id': admin['_id'],
                'username': admin['username'],
                'email': admin.get('email', ''),
                'role': admin.get('role', 'admin'),
                'created_at': admin.get('created_at', '').isoformat() if admin.get('created_at') else None
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ========================================
# MongoDB API Endpoints
# ========================================

@app.route('/api/violations', methods=['GET'])
def get_violations():
    """
    Get all violations with optional filtering
    Query params: limit, skip, type, status
    """
    try:
        limit = int(request.args.get('limit', 100))
        skip = int(request.args.get('skip', 0))
        violation_type = request.args.get('type')
        status = request.args.get('status')
        
        # Build filters
        filters = {}
        if violation_type:
            filters['type'] = violation_type
        if status:
            filters['status'] = status
        
        violations = ViolationModel.get_all_violations(
            limit=limit,
            skip=skip,
            filters=filters if filters else None
        )
        
        return jsonify({
            'success': True,
            'count': len(violations),
            'violations': violations
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/violations/<violation_id>', methods=['GET'])
def get_violation(violation_id):
    """Get a single violation by ID"""
    try:
        violation = ViolationModel.get_violation_by_id(violation_id)
        
        if violation:
            return jsonify({
                'success': True,
                'violation': violation
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Violation not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/violations/<violation_id>', methods=['PUT'])
def update_violation(violation_id):
    """Update a violation record"""
    try:
        update_data = request.json
        
        success = ViolationModel.update_violation(violation_id, update_data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Violation updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update violation'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/violations/<violation_id>', methods=['DELETE'])
def delete_violation(violation_id):
    """Delete a violation record"""
    try:
        success = ViolationModel.delete_violation(violation_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Violation deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to delete violation'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/violations/stats', methods=['GET'])
def get_violation_stats():
    """Get violation statistics"""
    try:
        stats = ViolationModel.get_violation_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get all cameras"""
    try:
        cameras = CameraModel.get_all_cameras()
        
        return jsonify({
            'success': True,
            'count': len(cameras),
            'cameras': cameras
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cameras', methods=['POST'])
def create_camera():
    """Create a new camera record"""
    try:
        camera_data = request.json
        
        camera_id = CameraModel.create_camera(camera_data)
        
        if camera_id:
            return jsonify({
                'success': True,
                'message': 'Camera created successfully',
                'camera_id': camera_id
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create camera'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check API and database health"""
    db_status = db.connected if db is not None else False
    
    response = {
        'api_status': 'running',
        'database_connected': db_status,
        'database_type': 'MongoDB' if db_status else 'None'
    }
    
    # Add database info if connected
    if db_status:
        db_info = get_database_info()
        if db_info:
            response['database_info'] = db_info
    
    return jsonify(response)


@app.route('/api/database/info', methods=['GET'])
def database_info():
    """Get detailed database information"""
    try:
        db_info = get_database_info()
        
        if db_info:
            return jsonify({
                'success': True,
                'database': db_info
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Database not connected'
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=False, port=5001)