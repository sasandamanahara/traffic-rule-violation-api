"""
API Blueprint initialization
"""
from flask import Blueprint

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Import all route modules to register them
from . import auth, violations, cameras, health, detection

