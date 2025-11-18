"""
Camera stream manager for live detection
Supports RTSP streams and USB/webcam devices
"""
import cv2
import time
import threading
import uuid
import os
from datetime import datetime
from ultralytics import YOLO
from config import Config
from db_models import ViolationModel, CameraModel
import sys

# Add parent directory to path to import from live folder
live_folder = os.path.join(os.path.dirname(__file__), '..', 'live')
if live_folder not in sys.path:
    sys.path.insert(0, live_folder)

try:
    from helmet_triple_detection import check_helmet_triple, model_helmet, model_triple
    print("✅ Loaded helmet_triple_detection from live folder")
except ImportError as e:
    print(f"⚠️ Warning: Could not import helmet_triple_detection from live folder: {e}")
    check_helmet_triple = None
    model_helmet = None
    model_triple = None

# Global stream storage
active_streams = {}
stream_lock = threading.Lock()

# Load models once
try:
    # Try new best.pt first (used in live folder), fallback to Vehical_Detection.pt
    try:
        vehicle_model = YOLO("models/new best.pt")
        print("✅ Vehicle model (new best.pt) loaded successfully")
    except:
        vehicle_model = YOLO("models/Vehical_Detection.pt")
        print("✅ Vehicle model (Vehical_Detection.pt) loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load vehicle model: {e}")
    vehicle_model = None

def open_camera(source_type, source):
    """
    Open camera capture based on source type
    
    Args:
        source_type: "rtsp" or "usb"
        source: RTSP URL or USB device index (string)
    
    Returns:
        tuple: (cv2.VideoCapture object or None, error_message or None)
    """
    error_messages = []
    
    try:
        if source_type == "rtsp":
            # Normalize RTSP URL - add default port and path if missing
            rtsp_url = source.strip()
            
            # If URL doesn't have a path, try common paths
            if rtsp_url.count('/') == 2:  # Only has rtsp://ip
                # Try common RTSP paths
                common_paths = ['/stream', '/live', '/h264', '/video', '/cam/realmonitor']
                for path in common_paths:
                    test_url = f"{rtsp_url}:554{path}"  # Default RTSP port is 554
                    print(f"[RTSP] Trying: {test_url}")
                    cap = cv2.VideoCapture(test_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
                    
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            print(f"[RTSP] Successfully connected to: {test_url}")
                            return (cap, None)
                        else:
                            error_messages.append(f"Could not read frame from {test_url}")
                        cap.release()
                    else:
                        error_messages.append(f"Could not open {test_url}")
                
                # If no path worked, try with just port
                test_url = f"{rtsp_url}:554"
                print(f"[RTSP] Trying: {test_url}")
                cap = cv2.VideoCapture(test_url)
            else:
                # URL has path, use as-is
                print(f"[RTSP] Connecting to: {rtsp_url}")
                # Try with GStreamer backend first (better RTSP support)
                try:
                    # GStreamer pipeline for RTSP (more reliable)
                    gst_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
                    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            print(f"[RTSP] Successfully connected using GStreamer: {rtsp_url}")
                            return (cap, None)
                        cap.release()
                except Exception as e:
                    print(f"[RTSP] GStreamer not available or failed: {e}")
                
                # Fallback to default backend
                cap = cv2.VideoCapture(rtsp_url)
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
        else:  # usb
            print(f"[USB] Opening device: {source}")
            cap = cv2.VideoCapture(int(source))
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
        
        # Test if camera opened successfully
        if not cap.isOpened():
            error_msg = f"Failed to open camera: {source}"
            if error_messages:
                error_msg += f"\nTried: {', '.join(error_messages)}"
            print(f"[ERROR] {error_msg}")
            return (None, error_msg)
        
        # Try to read a frame to verify connection
        print(f"[INFO] Attempting to read frame from {source}...")
        ret, frame = cap.read()
        if not ret:
            error_msg = f"Failed to read frame from camera: {source}. Camera may be busy or stream unavailable."
            print(f"[ERROR] {error_msg}")
            cap.release()
            return (None, error_msg)
        
        print(f"[SUCCESS] Camera opened and verified: {source}")
        return (cap, None)
    except Exception as e:
        error_msg = f"Exception opening camera: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return (None, error_msg)

def detect_violations_periodic(frame, camera_id, location):
    """
    Detect violations in a frame using periodic detection
    
    Args:
        frame: OpenCV frame
        camera_id: Camera ID for violation records
        location: Camera location string
    
    Returns:
        List of violation dictionaries
    """
    violations = []
    
    if not vehicle_model:
        return violations
    
    try:
        # Vehicle detection
        results = vehicle_model.track(frame, persist=True, conf=0.3, verbose=False)
        
        if not results or len(results) == 0 or results[0].boxes is None:
            print(f"[Detection] No vehicles detected in frame")
            return violations
        
        detected_vehicles = []
        motorcycles_found = 0
        riders_found = 0
        motor_boxes = []
        rider_boxes = []
        
        # First pass: collect motorcycles and riders (like live/vehicle_detection.py)
        for box in results[0].boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = vehicle_model.names[cls_id].lower()
            detected_vehicles.append(cls_name)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            
            # Check if it's a motorcycle
            if "motor" in cls_name or "bike" in cls_name or cls_id == 3:
                motorcycles_found += 1
                motor_boxes.append({
                    'box': box,
                    'xyxy': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'cls_name': cls_name
                })
            # Check if it's a rider (separate detection in some models)
            elif "rider" in cls_name or "person" in cls_name:
                riders_found += 1
                rider_boxes.append({
                    'box': box,
                    'xyxy': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'cls_name': cls_name
                })
        
        # Second pass: Process motorcycles (combine with riders if overlapping)
        for motor_data in motor_boxes:
            x1, y1, x2, y2 = motor_data['xyxy']
            confidence = motor_data['confidence']
            cls_name = motor_data['cls_name']
            
            # Check for overlapping riders (expand bbox if rider overlaps)
            expanded_bbox = [x1, y1, x2, y2]
            has_rider_overlap = False
            
            for rider_data in rider_boxes:
                rx1, ry1, rx2, ry2 = rider_data['xyxy']
                # Check if rider overlaps with motorcycle
                if (x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1):
                    expanded_bbox[0] = min(x1, rx1)
                    expanded_bbox[1] = min(y1, ry1)
                    expanded_bbox[2] = max(x2, rx2)
                    expanded_bbox[3] = max(y2, ry2)
                    has_rider_overlap = True
                    print(f"[Detection] Motorcycle with overlapping rider detected")
            
            # Use expanded bbox if rider overlaps, otherwise use motorcycle bbox
            final_x1, final_y1, final_x2, final_y2 = expanded_bbox
            
            # Ensure valid coordinates
            if final_x2 > final_x1 and final_y2 > final_y1 and final_x1 >= 0 and final_y1 >= 0:
                crop = frame[final_y1:final_y2, final_x1:final_x2]
                
                if crop.size > 0:
                    print(f"[Detection] Processing motorcycle: {cls_name} at [{final_x1},{final_y1},{final_x2},{final_y2}], confidence: {confidence:.2f}")
                    
                    # Check for violations using models directly
                    try:
                        if model_triple is None or model_helmet is None:
                            # Try to import again
                            from helmet_triple_detection import model_helmet, model_triple
                        
                        if model_triple is None or model_helmet is None:
                            print(f"[Detection] Warning: Helmet/Triple models not loaded")
                            continue
                        
                        # Check triple riding (lower confidence threshold for testing)
                        triple_results = model_triple(crop, conf=0.5, verbose=False)
                        if len(triple_results) > 0 and len(triple_results[0].boxes) > 0:
                            triple_conf = float(triple_results[0].boxes.conf[0].cpu().numpy())
                            print(f"[Detection] ✅ Triple Riding violation detected! Confidence: {triple_conf:.2f}")
                            violation = {
                                'type': 'Triple Riding',
                                'confidence': triple_conf,
                                'bbox': [final_x1, final_y1, final_x2, final_y2],
                                'camera_id': camera_id,
                                'location': location,
                                'frame': None,
                                'timestamp': time.time()
                            }
                            violations.append(violation)
                        
                        # Check helmet (check all detections, not just first)
                        helmet_results = model_helmet(crop, conf=0.3, verbose=False)
                        if len(helmet_results) > 0 and len(helmet_results[0].boxes) > 0:
                            helmet_classes = helmet_results[0].boxes.cls.cpu().numpy()
                            helmet_names = [helmet_results[0].names[int(b)].lower() for b in helmet_classes]
                            helmet_count = sum(1 for name in helmet_names if name == "helmet")
                            
                            print(f"[Detection] Helmet detection: {helmet_count} helmet(s) found, classes: {helmet_names}")
                            
                            if helmet_count == 0:  # No helmet detected
                                # Get highest confidence from all detections
                                helmet_confs = helmet_results[0].boxes.conf.cpu().numpy()
                                helmet_conf = float(max(helmet_confs)) if len(helmet_confs) > 0 else confidence
                                print(f"[Detection] ✅ NO HELMET violation detected! Confidence: {helmet_conf:.2f}")
                                violation = {
                                    'type': 'Helmet',
                                    'confidence': helmet_conf,
                                    'bbox': [final_x1, final_y1, final_x2, final_y2],
                                    'camera_id': camera_id,
                                    'location': location,
                                    'frame': None,
                                    'timestamp': time.time()
                                }
                                violations.append(violation)
                        else:
                            print(f"[Detection] No helmet detections in crop (may indicate no helmet violation)")
                            # If helmet model detects nothing, it might mean no helmet - create violation
                            # But only if we're confident there's a rider/motorcycle
                            if has_rider_overlap or confidence > 0.5:
                                print(f"[Detection] ✅ NO HELMET violation (no detections in crop with rider present)")
                                violation = {
                                    'type': 'Helmet',
                                    'confidence': confidence * 0.8,  # Slightly lower confidence
                                    'bbox': [final_x1, final_y1, final_x2, final_y2],
                                    'camera_id': camera_id,
                                    'location': location,
                                    'frame': None,
                                    'timestamp': time.time()
                                }
                                violations.append(violation)
                    except Exception as e:
                        print(f"[Detection] Error in helmet/triple detection: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[Detection] Invalid crop size for motorcycle at [{final_x1},{final_y1},{final_x2},{final_y2}]")
        
        if detected_vehicles:
            unique_vehicles = set(detected_vehicles)
            print(f"[Detection] Summary - Total detections: {len(detected_vehicles)}, Types: {unique_vehicles}")
            print(f"[Detection] Motorcycles: {motorcycles_found}, Riders: {riders_found}, Violations found: {len(violations)}")
            if motorcycles_found == 0:
                print(f"[Detection] ⚠️ No motorcycles detected. Vehicle model may not detect motorcycles, or none are in frame.")
                print(f"[Detection] Available vehicle classes in model: {list(vehicle_model.names.values())[:20]}...")  # Show first 20
        else:
            print(f"[Detection] No vehicles detected in frame")
    
    except Exception as e:
        print(f"Error in periodic detection: {e}")
        import traceback
        traceback.print_exc()
    
    return violations

def save_violation_to_db(violation, camera_id, frame=None):
    """
    Save violation to MongoDB and create snapshot
    
    Args:
        violation: Violation dictionary
        camera_id: Camera ID
        frame: OpenCV frame to save as snapshot (optional)
    """
    try:
        # Create snapshot
        os.makedirs(Config.SNAPSHOT_FOLDER, exist_ok=True)
        snapshot_filename = f"{uuid.uuid4()}.jpg"
        snapshot_path = os.path.join(Config.SNAPSHOT_FOLDER, snapshot_filename)
        
        # Save frame snapshot if provided
        if frame is not None:
            try:
                # Draw bounding box on frame
                x1, y1, x2, y2 = violation['bbox']
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_copy, f"{violation['type']} Violation", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(snapshot_path, frame_copy)
            except Exception as e:
                print(f"Error saving snapshot: {e}")
        
        snapshot_url = f"{Config.API_BASE_URL}/static/snapshots/{snapshot_filename}"
        
        # Prepare violation data
        violation_data = {
            'type': violation['type'],
            'confidence': violation['confidence'],
            'bbox': violation['bbox'],
            'snapshot_url': snapshot_url,
            'camera_id': camera_id,
            'status': 'pending',
            'location': violation.get('location', 'Unknown'),
            'description': f"{violation['type']} violation detected",
            'created_at': datetime.utcnow()
        }
        
        # Save to MongoDB
        violation_id = ViolationModel.create_violation(violation_data)
        if violation_id:
            violation['_id'] = violation_id
            violation['snapshot_url'] = snapshot_url
            print(f"✅ Saved {violation['type']} violation to database: {violation_id}")
        else:
            print(f"⚠️ Failed to save violation to database")
        
        return violation_id
    except Exception as e:
        print(f"Error saving violation to DB: {e}")
        import traceback
        traceback.print_exc()
        return None

def detection_thread(stream_id, camera_id, source_type, source, snapshot_interval):
    """
    Background thread for camera stream and periodic detection
    
    Args:
        stream_id: Unique stream ID
        camera_id: Camera ID from database
        source_type: "rtsp" or "usb"
        source: RTSP URL or USB device index
        snapshot_interval: Seconds between detections
    """
    global active_streams
    
    print(f"[Stream {stream_id}] Starting detection thread...")
    
    # Get camera info
    camera = CameraModel.get_camera_by_id(camera_id) if camera_id else None
    location = camera.get('location', 'Unknown') if camera else 'Unknown'
    
    # Open camera
    cap, error_msg = open_camera(source_type, source)
    if not cap:
        with stream_lock:
            if stream_id in active_streams:
                active_streams[stream_id]['status'] = 'error'
                active_streams[stream_id]['error'] = error_msg or 'Failed to open camera'
        print(f"[Stream {stream_id}] Failed to open camera: {error_msg}")
        return
    
    # Initialize stream data
    with stream_lock:
        active_streams[stream_id]['cap'] = cap
        active_streams[stream_id]['status'] = 'active'
        active_streams[stream_id]['started_at'] = time.time()
        active_streams[stream_id]['frame_count'] = 0
        active_streams[stream_id]['violation_count'] = 0
    
    last_detection_time = time.time()
    frame_count = 0
    
    print(f"[Stream {stream_id}] Camera opened successfully, starting detection loop...")
    
    try:
        while True:
            with stream_lock:
                if stream_id not in active_streams or active_streams[stream_id]['status'] == 'stopping':
                    break
            
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                # Check if stream should continue
                with stream_lock:
                    if stream_id not in active_streams:
                        break
                continue
            
            frame_count += 1
            
            # Store latest frame for MJPEG stream
            with stream_lock:
                if stream_id in active_streams:
                    active_streams[stream_id]['last_frame'] = frame.copy()
                    active_streams[stream_id]['frame_count'] = frame_count
                    # Debug: log every 100 frames
                    if frame_count % 100 == 0:
                        print(f"[Stream {stream_id}] Frame {frame_count} stored for MJPEG stream")
            
            # Periodic detection every snapshot_interval seconds
            current_time = time.time()
            if current_time - last_detection_time >= snapshot_interval:
                print(f"[Stream {stream_id}] Running periodic detection (frame {frame_count})...")
                
                # Detect violations
                violations = detect_violations_periodic(frame, camera_id, location)
                
                print(f"[Stream {stream_id}] Detection complete: {len(violations)} violation(s) found")
                
                # Save violations to database
                saved_count = 0
                for violation in violations:
                    violation_id = save_violation_to_db(violation, camera_id, frame)
                    if violation_id:
                        violation['_id'] = violation_id
                        saved_count += 1
                        # Add to violations queue
                        with stream_lock:
                            if stream_id in active_streams:
                                active_streams[stream_id]['violations_queue'].append(violation)
                                active_streams[stream_id]['violation_count'] += 1
                        print(f"[Stream {stream_id}] ✅ Saved {violation['type']} violation: {violation_id}")
                    else:
                        print(f"[Stream {stream_id}] ⚠️ Failed to save {violation['type']} violation")
                
                if saved_count > 0:
                    print(f"[Stream {stream_id}] Successfully saved {saved_count}/{len(violations)} violations to database")
                
                last_detection_time = current_time
            
            # Small delay to prevent CPU overload
            time.sleep(0.03)  # ~30 FPS max
    
    except Exception as e:
        print(f"[Stream {stream_id}] Error in detection thread: {e}")
        import traceback
        traceback.print_exc()
        with stream_lock:
            if stream_id in active_streams:
                active_streams[stream_id]['status'] = 'error'
                active_streams[stream_id]['error'] = str(e)
    finally:
        # Cleanup
        cap.release()
        with stream_lock:
            if stream_id in active_streams:
                active_streams[stream_id]['status'] = 'stopped'
                active_streams[stream_id]['cap'] = None
        print(f"[Stream {stream_id}] Detection thread stopped")

def start_stream(camera_id, source_type, source, snapshot_interval=5):
    """
    Start a camera stream
    
    Args:
        camera_id: Camera ID from database
        source_type: "rtsp" or "usb"
        source: RTSP URL or USB device index (string)
        snapshot_interval: Seconds between detections (default: 5)
    
    Returns:
        stream_id (str) or None if failed
    """
    global active_streams
    
    stream_id = str(uuid.uuid4())
    
    # Initialize stream data
    with stream_lock:
        active_streams[stream_id] = {
            'camera_id': camera_id,
            'source_type': source_type,
            'source': source,
            'status': 'starting',
            'cap': None,
            'thread': None,
            'last_frame': None,
            'violations_queue': [],
            'started_at': None,
            'frame_count': 0,
            'violation_count': 0,
            'error': None
        }
    
    # Start detection thread
    thread = threading.Thread(
        target=detection_thread,
        args=(stream_id, camera_id, source_type, source, snapshot_interval),
        daemon=True
    )
    thread.start()
    
    with stream_lock:
        active_streams[stream_id]['thread'] = thread
    
    # Update camera status
    if camera_id:
        CameraModel.update_camera(camera_id, {
            'stream_id': stream_id,
            'last_stream_started': datetime.utcnow()
        })
    
    print(f"✅ Started stream {stream_id} for camera {camera_id}")
    return stream_id

def stop_stream(stream_id):
    """
    Stop a camera stream
    
    Args:
        stream_id: Stream ID to stop
    
    Returns:
        bool: True if successful
    """
    global active_streams
    
    with stream_lock:
        if stream_id not in active_streams:
            return False
        
        stream_data = active_streams[stream_id]
        
        # Mark as stopping
        stream_data['status'] = 'stopping'
        
        # Release camera
        if stream_data['cap']:
            try:
                stream_data['cap'].release()
            except:
                pass
        
        # Get camera_id before removing
        camera_id = stream_data.get('camera_id')
    
    # Wait a bit for thread to finish
    time.sleep(0.5)
    
    # Remove from active streams
    with stream_lock:
        if stream_id in active_streams:
            del active_streams[stream_id]
    
    # Update camera status
    if camera_id:
        CameraModel.update_camera(camera_id, {
            'stream_id': None
        })
    
    print(f"✅ Stopped stream {stream_id}")
    return True

def get_stream_status(stream_id):
    """
    Get status of a stream
    
    Args:
        stream_id: Stream ID
    
    Returns:
        dict: Stream status or None if not found
    """
    global active_streams
    
    with stream_lock:
        if stream_id not in active_streams:
            return None
        
        stream_data = active_streams[stream_id].copy()
        
        # Calculate uptime
        if stream_data.get('started_at'):
            uptime = time.time() - stream_data['started_at']
        else:
            uptime = 0
        
        # Calculate FPS (approximate)
        fps = 0
        if uptime > 0 and stream_data.get('frame_count', 0) > 0:
            fps = stream_data['frame_count'] / uptime
        
        return {
            'stream_id': stream_id,
            'camera_id': stream_data.get('camera_id'),
            'status': stream_data.get('status', 'unknown'),
            'source_type': stream_data.get('source_type'),
            'source': stream_data.get('source'),
            'uptime': round(uptime, 2),
            'frame_count': stream_data.get('frame_count', 0),
            'violation_count': stream_data.get('violation_count', 0),
            'fps': round(fps, 2),
            'error': stream_data.get('error')
        }

def get_latest_frame(stream_id):
    """
    Get latest frame from stream for MJPEG
    
    Args:
        stream_id: Stream ID
    
    Returns:
        numpy array (frame) or None
    """
    global active_streams
    
    with stream_lock:
        if stream_id not in active_streams:
            return None
        frame = active_streams[stream_id].get('last_frame')
        # Debug: log when frame is retrieved (every 30 calls to avoid spam)
        if frame is not None and hasattr(get_latest_frame, '_call_count'):
            get_latest_frame._call_count = (get_latest_frame._call_count + 1) % 30
            if get_latest_frame._call_count == 0:
                print(f"[Stream {stream_id}] Frame retrieved for MJPEG (frame exists)")
        elif frame is None and hasattr(get_latest_frame, '_call_count'):
            get_latest_frame._call_count = (get_latest_frame._call_count + 1) % 30
            if get_latest_frame._call_count == 0:
                print(f"[Stream {stream_id}] No frame available for MJPEG")
        else:
            get_latest_frame._call_count = 0
        return frame

def get_recent_violations(stream_id, limit=20):
    """
    Get recent violations from stream queue
    
    Args:
        stream_id: Stream ID
        limit: Maximum number of violations to return
    
    Returns:
        list: Recent violations
    """
    global active_streams
    
    with stream_lock:
        if stream_id not in active_streams:
            return []
        
        violations = active_streams[stream_id]['violations_queue'][-limit:]
        return violations

