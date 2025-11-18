from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from config import Config
from database import get_db
from api import api_bp

app = Flask(__name__)

# Configure app from config
app.config['SECRET_KEY'] = Config.SECRET_KEY

# Configure CORS
CORS(app, 
     origins=Config.CORS_ORIGINS,
     methods=Config.CORS_METHODS,
     allow_headers=Config.CORS_ALLOW_HEADERS,
     expose_headers=Config.CORS_EXPOSE_HEADERS,
     supports_credentials=Config.CORS_SUPPORTS_CREDENTIALS)

# Initialize database connection
db = get_db()

# Register API blueprint
app.register_blueprint(api_bp)

# Static file routes
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join('static', filename), mimetype='image/jpeg')

@app.route('/output_videos/<path:filename>')
def serve_processed_video(filename):
    return send_file(os.path.join('output_videos', filename), mimetype='video/mp4')

@app.route('/processed/<path:filename>')
def serve_video(filename):
    return send_file(os.path.join('processed', filename), mimetype='video/mp4', as_attachment=False)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(debug=Config.FLASK_DEBUG, port=port)
