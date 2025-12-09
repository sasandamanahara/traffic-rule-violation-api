"""
Detection and prediction API routes
"""
from flask import request, jsonify
from . import api_bp
from config import Config
import os
import uuid
from detect_seperate_violations.helmet_triple_processor import detect_helmet_triple_in_video
from detect_seperate_violations.redlight_violation_processor import detect_redlight_violation_in_video


@api_bp.route('/process-video-helmet', methods=['POST'])
def process_video_route_helmet():
    """Process video only for helmet + triple riding"""

    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        file_id = f"{uuid.uuid4()}.mp4"
        input_path = os.path.join(Config.UPLOAD_FOLDER, file_id)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(input_path)

        result_json = detect_helmet_triple_in_video(
            input_video_path=input_path,
            snapshot_folder=Config.SNAPSHOT_FOLDER,
            output_video_folder=Config.OUTPUT_VIDEO_FOLDER
        )

        result_data = result_json.get_json()  # now this is a Python dict

        # Access values individually
        total_frames = result_data['totalFrames']
        processed_frames = result_data['processedFrames']
        processing_time = result_data['processingTime']
        violations = result_data['violations']

        print(f"Total Frames: {total_frames}")
        print(f"Processed Frames: {processed_frames}")
        print(f"Processing Time: {processing_time} seconds")
        print(f"Violations Detected: {len(violations)}")

        # You can now use these values or return them directly
        return jsonify({
            "totalFrames": total_frames,
            "processedFrames": processed_frames,
            "processingTime": processing_time,
            "violationsDetected": len(violations),
            "violations": violations
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500








@api_bp.route('/process-video-trafficlight', methods=['POST'])
def process_video_route_trafficlight():
    """Process video only for traffic light violation"""

    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        file_id = f"{uuid.uuid4()}.mp4"
        input_path = os.path.join(Config.UPLOAD_FOLDER, file_id)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(input_path)

        result_json = detect_redlight_violation_in_video(
            input_video_path=input_path,
            snapshot_folder=Config.SNAPSHOT_FOLDER,
            output_video_folder=Config.OUTPUT_VIDEO_FOLDER
        )

        result_data = result_json.get_json()  # now this is a Python dict

        # Access values individually
        total_frames = result_data['totalFrames']
        processed_frames = result_data['processedFrames']
        processing_time = result_data['processingTime']
        violations = result_data['violations']

        print(f"Total Frames: {total_frames}")
        print(f"Processed Frames: {processed_frames}")
        print(f"Processing Time: {processing_time} seconds")
        print(f"Violations Detected: {len(violations)}")

        # You can now use these values or return them directly
        return jsonify({
            "totalFrames": total_frames,
            "processedFrames": processed_frames,
            "processingTime": processing_time,
            "violationsDetected": len(violations),
            "violations": violations
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500
    


# @api_bp.route('/lanes', methods=['POST'])
# def save_lanes():
#     """Process video with lane detection"""
#     if 'video' not in request.files or 'pixels_file' not in request.files:
#         return jsonify({'error': 'Video or lane data file missing'}), 400

#     video_file = request.files['video']
#     pixels_file = request.files['pixels_file']

#     # Save video
#     os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
#     video_path = os.path.join(Config.UPLOAD_FOLDER, video_file.filename)
#     lane_json_path = os.path.join(Config.UPLOAD_FOLDER, pixels_file.filename)
#     video_file.save(video_path)
#     pixels_file.save(lane_json_path)

#     try:
#         output_video, violations_json = process_video_with_lanes(video_path, lane_json_path)
#         return jsonify({
#             "message": "Processing completed",
#             "output_video": output_video,
#             "violations": violations_json
#         }), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

