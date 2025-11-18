"""
Live camera detection API routes
"""
from flask import request, jsonify, Response
from . import api_bp
from .camera_streams import (
    start_stream, stop_stream, get_stream_status,
    get_latest_frame, get_recent_violations
)
import cv2
import time

@api_bp.route('/live/start', methods=['POST'])
def start_live_stream():
    """
    Start a live camera stream
    
    Request body:
    {
        "camera_id": "camera_id_string",
        "source_type": "rtsp" | "usb",
        "source": "rtsp://..." or "0",
        "snapshot_interval": 5
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        camera_id = data.get('camera_id')
        source_type = data.get('source_type')
        source = data.get('source')
        snapshot_interval = data.get('snapshot_interval', 5)
        
        # Validate required fields
        if not source_type or source_type not in ['rtsp', 'usb']:
            return jsonify({
                'success': False,
                'error': 'source_type must be "rtsp" or "usb"'
            }), 400
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'source is required'
            }), 400
        
        # Validate snapshot_interval
        try:
            snapshot_interval = float(snapshot_interval)
            if snapshot_interval < 1:
                snapshot_interval = 1
            elif snapshot_interval > 60:
                snapshot_interval = 60
        except (ValueError, TypeError):
            snapshot_interval = 5
        
        # Start stream
        stream_id = start_stream(camera_id, source_type, source, snapshot_interval)
        
        if stream_id:
            return jsonify({
                'success': True,
                'stream_id': stream_id,
                'status': 'started',
                'message': 'Stream started successfully'
            }), 200
        else:
            error_msg = 'Failed to start stream'
            if source_type == 'rtsp':
                error_msg += '. Please check:\n'
                error_msg += '1. RTSP URL format: rtsp://[username:password@]ip:port/path\n'
                error_msg += '2. Common formats:\n'
                error_msg += '   - rtsp://192.168.1.6:554/stream\n'
                error_msg += '   - rtsp://admin:password@192.168.1.6:554/live\n'
                error_msg += '   - rtsp://192.168.1.6:554/h264\n'
                error_msg += '3. Camera is accessible on the network\n'
                error_msg += '4. Firewall allows RTSP connections (port 554)'
            
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/live/stop/<stream_id>', methods=['POST'])
def stop_live_stream(stream_id):
    """
    Stop a live camera stream
    
    Args:
        stream_id: Stream ID to stop
    """
    try:
        success = stop_stream(stream_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Stream stopped successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Stream not found or already stopped'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/live/status/<stream_id>', methods=['GET'])
def get_live_stream_status(stream_id):
    """
    Get status of a live camera stream
    
    Args:
        stream_id: Stream ID
    """
    try:
        status = get_stream_status(stream_id)
        
        if status:
            return jsonify({
                'success': True,
                'status': status
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Stream not found'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/live/stream/<stream_id>')
def stream_video(stream_id):
    """
    MJPEG video stream endpoint
    
    Args:
        stream_id: Stream ID
    """
    def generate():
        import cv2
        consecutive_none_count = 0
        max_none_count = 100  # Stop after 100 consecutive None frames (10 seconds at 0.1s delay)
        
        while True:
            try:
                frame = get_latest_frame(stream_id)
                
                if frame is None:
                    consecutive_none_count += 1
                    if consecutive_none_count > max_none_count:
                        print(f"[Stream {stream_id}] No frames available, stopping stream")
                        break
                    # Send a black placeholder frame
                    placeholder = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
                    cv2.putText(placeholder, "Waiting for frames...", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(0.1)
                    continue
                
                # Reset consecutive none count when we get a frame
                consecutive_none_count = 0
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    print(f"[Stream {stream_id}] Failed to encode frame")
                    time.sleep(0.1)
            except Exception as e:
                print(f"[Stream {stream_id}] Error in stream generator: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
            
            # Small delay to control frame rate (~30 FPS)
            time.sleep(0.033)
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )

@api_bp.route('/live/violations/<stream_id>', methods=['GET'])
def get_live_violations(stream_id):
    """
    Get recent violations from a live stream
    
    Args:
        stream_id: Stream ID
    Query params:
        limit: Maximum number of violations (default: 20)
    """
    try:
        limit = int(request.args.get('limit', 20))
        if limit < 1:
            limit = 1
        elif limit > 100:
            limit = 100
        
        violations = get_recent_violations(stream_id, limit)
        
        return jsonify({
            'success': True,
            'count': len(violations),
            'violations': violations
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

