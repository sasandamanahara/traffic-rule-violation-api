"""
Authentication API routes
"""
from flask import request, jsonify
from . import api_bp
from db_models import AdminModel
from .utils import hash_password, verify_password, generate_token, verify_token, require_auth

# Handle OPTIONS preflight requests
@api_bp.route('/auth/register', methods=['OPTIONS'])
def register_options():
    return '', 200

@api_bp.route('/auth/register', methods=['POST'])
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


@api_bp.route('/auth/login', methods=['POST'])
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


@api_bp.route('/auth/verify', methods=['POST'])
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


@api_bp.route('/auth/me', methods=['GET'])
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




