"""
Live detection API endpoints
"""
import time
import cv2
from flask import request, jsonify, Response
from . import api_bp
from live_detection_service import get_service
from .utils import require_auth


@api_bp.route('/live/start', methods=['POST'])
@require_auth
def start_live_detection():
    """Start live detection"""
    try:
        data = request.json or {}
        video_source = data.get('video_source')
        violation_types = data.get('violation_types', [
            'helmet', 'triple_riding', 'speed', 'direction', 'red_light', 'no_parking'
        ])
        
        if not video_source:
            return jsonify({
                'success': False,
                'error': 'video_source is required'
            }), 400
        
        service = get_service()
        
        # Check if already running
        if service.running:
            return jsonify({
                'success': False,
                'error': 'Detection is already running. Stop it first.'
            }), 400
        
        # Start detection
        try:
            service.start(video_source, violation_types)
        except RuntimeError as e:
            error_msg = str(e)
            error_type = "unknown"
            
            # Categorize and enhance error messages
            if 'Calibration failed' in error_msg:
                error_type = "calibration_error"
                # Determine protocol type
                is_rtmp = 'rtmp://' in video_source
                is_rtsp = 'rtsp://' in video_source
                is_file = video_source.endswith(('.mp4', '.avi', '.mov', '.m4v', '.mkv', '.flv'))
                
                if is_rtmp or is_rtsp:
                    protocol = 'RTMP' if is_rtmp else 'RTSP'
                    error_msg = f"{error_msg}\n\nTroubleshooting tips for {protocol} stream:"
                    error_msg += "\n- Ensure the stream server is running and accessible"
                    error_msg += "\n- Verify the stream is actively publishing"
                    error_msg += "\n- Check network connectivity and firewall settings"
                    error_msg += f"\n- Test the stream URL with: ffplay {video_source}"
                    if is_rtmp:
                        error_msg += "\n- For RTMP: Check if nginx-rtmp or OBS is running"
                        error_msg += "\n- Verify port 1935 is accessible"
                    elif is_rtsp:
                        error_msg += "\n- For RTSP: Check if the RTSP server is running"
                        error_msg += "\n- Verify the RTSP URL format is correct (e.g., rtsp://user:pass@host:port/stream)"
                        error_msg += "\n- Check if port 554 (default RTSP port) is accessible"
                elif is_file:
                    error_msg = f"{error_msg}\n\nTroubleshooting tips for video file:"
                    error_msg += "\n- Verify the file path is correct"
                    error_msg += "\n- Check if the file exists and is readable"
                    error_msg += "\n- Ensure the file is a valid video format"
                else:
                    error_msg = f"{error_msg}\n\nTroubleshooting tips:"
                    error_msg += "\n- Verify the video source URL or path is correct"
                    error_msg += "\n- Check if the source is accessible"
            elif 'connection' in error_msg.lower() or 'connect' in error_msg.lower():
                error_type = "connection_error"
                if 'rtmp://' in video_source or 'rtsp://' in video_source:
                    protocol = 'RTMP' if 'rtmp://' in video_source else 'RTSP'
                    error_msg = f"Failed to connect to {protocol} stream: {video_source}\n\n"
                    error_msg += f"Possible causes:\n"
                    error_msg += f"- {protocol} server is not running\n"
                    error_msg += f"- Stream is not actively publishing\n"
                    error_msg += f"- Network connectivity issues\n"
                    error_msg += f"- Incorrect stream URL\n"
                    error_msg += f"\nTry testing with: ffplay {video_source}"
            elif 'not available' in error_msg.lower():
                error_type = "module_error"
                error_msg = f"Module import error: {error_msg}\n\n"
                error_msg += "This may indicate a missing dependency or import path issue."
            
            print(f"[ERROR] Detection start failed ({error_type}): {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'error_type': error_type,
                'video_source': video_source
            }), 500
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] Unexpected error starting detection: {error_details}")
            error_msg = f"Unexpected error: {str(e)}"
            if 'rtmp://' in video_source or 'rtsp://' in video_source:
                error_msg += f"\n\nStream URL: {video_source}"
                error_msg += "\nThis may be a stream connection or format issue."
            return jsonify({
                'success': False,
                'error': error_msg,
                'error_type': 'unexpected_error'
            }), 500
        
        return jsonify({
            'success': True,
            'status': service.get_status()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/live/stop', methods=['POST'])
@require_auth
def stop_live_detection():
    """Stop live detection"""
    try:
        service = get_service()
        
        if not service.running:
            return jsonify({
                'success': False,
                'error': 'Detection is not currently running'
            }), 400
        
        service.stop()
        
        return jsonify({
            'success': True,
            'message': 'Detection stopped successfully'
        })
    
    except Exception as e:
        import traceback
        print(f"[ERROR] Error stopping detection: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Failed to stop detection: {str(e)}"
        }), 500


@api_bp.route('/live/status', methods=['GET'])
@require_auth
def get_live_status():
    """Get live detection status"""
    try:
        service = get_service()
        status = service.get_status()
        
        # Add connection health information
        if status.get('running'):
            if not status.get('connected'):
                status['connection_warning'] = 'Stream may be disconnected'
        
        return jsonify({
            'success': True,
            'status': status
        })
    
    except Exception as e:
        import traceback
        print(f"[ERROR] Error getting status: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Failed to get status: {str(e)}"
        }), 500


@api_bp.route('/live/violations', methods=['GET'])
@require_auth
def get_live_violations():
    """Get recent violations from live detection"""
    try:
        limit = int(request.args.get('limit', 20))
        limit = min(limit, 100)  # Max 100
        
        service = get_service()
        
        if not service.running:
            return jsonify({
                'success': False,
                'error': 'Detection is not currently running'
            }), 400
        
        violations = service.get_violations(limit)
        
        return jsonify({
            'success': True,
            'violations': violations,
            'count': len(violations),
            'limit': limit
        })
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f"Invalid limit parameter: {str(e)}"
        }), 400
    except Exception as e:
        import traceback
        print(f"[ERROR] Error getting violations: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Failed to get violations: {str(e)}"
        }), 500


@api_bp.route('/live/stream', methods=['GET'])
def get_live_stream():
    """Get MJPEG video stream"""
    # Check authentication via token in query param or header
    token = request.args.get('token')
    if not token:
        # Try header
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
    
    if not token:
        return jsonify({
            'success': False,
            'error': 'Authentication token is missing'
        }), 401
    
    # Verify token
    from .utils import verify_token
    payload = verify_token(token)
    if not payload:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired token'
        }), 401
    
    def generate():
        service = get_service()
        
        while True:
            frame = service.get_frame()
            
            if frame is None:
                # Send a blank frame or wait
                time.sleep(0.1)
                continue
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
