"""
Violations API routes
"""
from flask import request, jsonify
from . import api_bp
from db_models import ViolationModel

@api_bp.route('/violations', methods=['GET'])
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


@api_bp.route('/violations/<violation_id>', methods=['GET'])
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


@api_bp.route('/violations/<violation_id>', methods=['PUT'])
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


@api_bp.route('/violations/<violation_id>', methods=['DELETE'])
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


@api_bp.route('/violations/stats', methods=['GET'])
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


