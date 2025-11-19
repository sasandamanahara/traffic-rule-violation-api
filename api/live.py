"""
Live Detection API routes
"""
import sys
import os
import base64
import io
from flask import jsonify, request, Response
from . import api_bp

# Add live folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'live'))
from live_detection_service import get_detection_service
from initialize import initialize_stream
from check_direction_initialize import track_vehicle_line_order


@api_bp.route('/live/start', methods=['POST'])
def start_live_detection():
    """Start live detection with video source"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        video_source = data.get('video_source')
        if not video_source:
            return jsonify({'error': 'video_source is required'}), 400
        
        # Check if detection is already running
        service = get_detection_service()
        if service.is_running:
            return jsonify({'error': 'Detection already running'}), 400
        
        # Initialize calibration
        print(f"[API] Initializing calibration for {video_source}...")
        calibration = initialize_stream(video_source)
        
        if not calibration:
            error_msg = f'Failed to connect to video source: {video_source}. Please verify the RTSP URL is correct and the stream is accessible.'
            return jsonify({'error': error_msg}), 500
        
        # Get vehicle direction data
        print("[API] Tracking vehicle directions...")
        vehicle_data = track_vehicle_line_order(video_source, calibration)
        vehicle_directions = vehicle_data.get('direction')
        
        # Start detection
        result = service.start_detection(video_source, calibration, vehicle_directions)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            'message': 'Live detection started',
            'status': service.get_status()
        })
    
    except Exception as e:
        print(f"[ERROR] Error starting detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/live/stop', methods=['POST'])
def stop_live_detection():
    """Stop live detection"""
    try:
        service = get_detection_service()
        result = service.stop_detection()
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            'message': 'Live detection stopped',
            'status': service.get_status()
        })
    
    except Exception as e:
        print(f"[ERROR] Error stopping detection: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/live/stream', methods=['GET'])
def live_stream():
    """MJPEG video stream endpoint"""
    def generate():
        service = get_detection_service()
        
        while True:
            frame = service.get_latest_frame()
            
            if frame is not None:
                # Encode frame as JPEG
                import cv2
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to prevent CPU spinning
            import time
            time.sleep(0.033)  # ~30 FPS
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@api_bp.route('/live/violations', methods=['GET'])
def get_live_violations():
    """Get recent violations from live detection"""
    try:
        service = get_detection_service()
        limit = request.args.get('limit', 50, type=int)
        
        violations = service.get_violations(limit=limit)
        
        # Convert image data to base64 for JSON response
        violations_json = []
        for v in violations:
            violation_data = {
                'id': v['id'],
                'type': v['type'],
                'vehicle_id': v['vehicle_id'],
                'timestamp': v['timestamp'],
                'bbox': v['bbox'],
                'metadata': v['metadata']
            }
            
            # Add base64 encoded image if available
            if 'image_data' in v:
                img_base64 = base64.b64encode(v['image_data']).decode('utf-8')
                violation_data['image'] = f"data:image/jpeg;base64,{img_base64}"
            
            violations_json.append(violation_data)
        
        return jsonify({
            'success': True,
            'violations': violations_json,
            'count': len(violations_json)
        })
    
    except Exception as e:
        print(f"[ERROR] Error getting violations: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/live/status', methods=['GET'])
def get_status():
    """Get current detection status"""
    try:
        service = get_detection_service()
        status = service.get_status()
        
        return jsonify({
            'success': True,
            'status': status
        })
    
    except Exception as e:
        print(f"[ERROR] Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

