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


@api_bp.route('/cameras/<camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Get a single camera by ID"""
    try:
        camera = CameraModel.get_camera_by_id(camera_id)
        
        if camera:
            return jsonify({
                'success': True,
                'camera': camera
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/cameras/<camera_id>', methods=['PUT'])
def update_camera(camera_id):
    """Update a camera record"""
    try:
        update_data = request.json
        
        success = CameraModel.update_camera(camera_id, update_data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Camera updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update camera'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/cameras/<camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Delete a camera record"""
    try:
        success = CameraModel.delete_camera(camera_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Camera deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to delete camera'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

