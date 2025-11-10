"""
Cameras API routes
"""
from flask import request, jsonify
from . import api_bp
from db_models import CameraModel

@api_bp.route('/cameras', methods=['GET'])
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


@api_bp.route('/cameras', methods=['POST'])
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

