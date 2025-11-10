"""
Health check API routes
"""
from flask import jsonify
from . import api_bp
from database import get_db, get_database_info

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Check API and database health"""
    db = get_db()
    db_status = db.connected
    
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


@api_bp.route('/database/info', methods=['GET'])
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

