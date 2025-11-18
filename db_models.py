"""
MongoDB models and data access layer for traffic violations
"""
from datetime import datetime
from bson import ObjectId
from database import get_collection

class ViolationModel:
    """Model for traffic violation records"""
    
    COLLECTION_NAME = 'violations'
    
    @staticmethod
    def create_violation(violation_data):
        """
        Create a new violation record
        
        Args:
            violation_data (dict): Violation information
            
        Returns:
            str: Inserted document ID or None if MongoDB not available
        """
        collection = get_collection(ViolationModel.COLLECTION_NAME)
        if collection is None:
            return None
        
        # Add timestamp if not present
        if 'created_at' not in violation_data:
            violation_data['created_at'] = datetime.utcnow()
        
        # Set default status if not present
        if 'status' not in violation_data:
            violation_data['status'] = 'pending'
        
        try:
            result = collection.insert_one(violation_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error creating violation: {str(e)}")
            return None
    
    @staticmethod
    def create_many_violations(violations_list):
        """
        Create multiple violation records at once
        
        Args:
            violations_list (list): List of violation dictionaries
            
        Returns:
            list: List of inserted document IDs
        """
        collection = get_collection(ViolationModel.COLLECTION_NAME)
        if collection is None or not violations_list:
            return []
        
        # Add timestamps and default status
        for violation in violations_list:
            if 'created_at' not in violation:
                violation['created_at'] = datetime.utcnow()
            if 'status' not in violation:
                violation['status'] = 'pending'
        
        try:
            result = collection.insert_many(violations_list)
            return [str(id) for id in result.inserted_ids]
        except Exception as e:
            print(f"Error creating violations: {str(e)}")
            return []
    
    @staticmethod
    def get_all_violations(limit=100, skip=0, filters=None):
        """
        Retrieve all violations with optional filtering
        
        Args:
            limit (int): Maximum number of records to return
            skip (int): Number of records to skip
            filters (dict): MongoDB query filters
            
        Returns:
            list: List of violation documents
        """
        collection = get_collection(ViolationModel.COLLECTION_NAME)
        if collection is None:
            return []
        
        query = filters if filters else {}
        
        try:
            violations = list(
                collection.find(query)
                .sort('created_at', -1)
                .skip(skip)
                .limit(limit)
            )
            
            # Convert ObjectId to string
            for violation in violations:
                violation['_id'] = str(violation['_id'])
            
            return violations
        except Exception as e:
            print(f"Error retrieving violations: {str(e)}")
            return []
    
    @staticmethod
    def get_violation_by_id(violation_id):
        """
        Get a single violation by ID
        
        Args:
            violation_id (str): Violation document ID
            
        Returns:
            dict: Violation document or None
        """
        collection = get_collection(ViolationModel.COLLECTION_NAME)
        if collection is None:
            return None
        
        try:
            violation = collection.find_one({'_id': ObjectId(violation_id)})
            if violation:
                violation['_id'] = str(violation['_id'])
            return violation
        except Exception as e:
            print(f"Error retrieving violation: {str(e)}")
            return None
    
    @staticmethod
    def update_violation(violation_id, update_data):
        """
        Update a violation record
        
        Args:
            violation_id (str): Violation document ID
            update_data (dict): Fields to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        collection = get_collection(ViolationModel.COLLECTION_NAME)
        if collection is None:
            return False
        
        update_data['updated_at'] = datetime.utcnow()
        
        try:
            result = collection.update_one(
                {'_id': ObjectId(violation_id)},
                {'$set': update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating violation: {str(e)}")
            return False
    
    @staticmethod
    def delete_violation(violation_id):
        """
        Delete a violation record
        
        Args:
            violation_id (str): Violation document ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        collection = get_collection(ViolationModel.COLLECTION_NAME)
        if collection is None:
            return False
        
        try:
            result = collection.delete_one({'_id': ObjectId(violation_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting violation: {str(e)}")
            return False
    
    @staticmethod
    def get_violations_by_type(violation_type, limit=50):
        """
        Get violations by type
        
        Args:
            violation_type (str): Type of violation (e.g., 'Helmet', 'Triple Riding')
            limit (int): Maximum number of records
            
        Returns:
            list: List of violations
        """
        return ViolationModel.get_all_violations(
            limit=limit,
            filters={'type': violation_type}
        )
    
    @staticmethod
    def get_violation_stats():
        """
        Get statistical summary of violations
        
        Returns:
            dict: Statistics dictionary
        """
        collection = get_collection(ViolationModel.COLLECTION_NAME)
        if collection is None:
            return {
                'total_violations': 0,
                'by_type': {},
                'by_status': {},
                'recent_count': 0
            }
        
        try:
            total = collection.count_documents({})
            
            # Count by type
            by_type = {}
            type_pipeline = [
                {'$group': {'_id': '$type', 'count': {'$sum': 1}}}
            ]
            for result in collection.aggregate(type_pipeline):
                by_type[result['_id']] = result['count']
            
            # Count by status
            by_status = {}
            status_pipeline = [
                {'$group': {'_id': '$status', 'count': {'$sum': 1}}}
            ]
            for result in collection.aggregate(status_pipeline):
                by_status[result['_id']] = result['count']
            
            # Recent violations (last 24 hours)
            from datetime import timedelta
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_count = collection.count_documents({
                'created_at': {'$gte': yesterday}
            })
            
            return {
                'total_violations': total,
                'by_type': by_type,
                'by_status': by_status,
                'recent_count': recent_count
            }
        except Exception as e:
            print(f"Error getting violation stats: {str(e)}")
            return {
                'total_violations': 0,
                'by_type': {},
                'by_status': {},
                'recent_count': 0
            }


class CameraModel:
    """Model for camera records"""
    
    COLLECTION_NAME = 'cameras'
    
    @staticmethod
    def create_camera(camera_data):
        """
        Create a new camera record
        
        Args:
            camera_data (dict): Camera information
            
        Returns:
            str: Inserted document ID or None
        """
        collection = get_collection(CameraModel.COLLECTION_NAME)
        if collection is None:
            return None
        
        camera_data['created_at'] = datetime.utcnow()
        camera_data['status'] = camera_data.get('status', 'active')
        
        try:
            result = collection.insert_one(camera_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error creating camera: {str(e)}")
            return None
    
    @staticmethod
    def get_all_cameras():
        """
        Get all camera records
        
        Returns:
            list: List of camera documents
        """
        collection = get_collection(CameraModel.COLLECTION_NAME)
        if collection is None:
            return []
        
        try:
            cameras = list(collection.find({}))
            for camera in cameras:
                camera['_id'] = str(camera['_id'])
            return cameras
        except Exception as e:
            print(f"Error retrieving cameras: {str(e)}")
            return []
    
    @staticmethod
    def get_active_cameras():
        """
        Get all active cameras
        
        Returns:
            list: List of active camera documents
        """
        collection = get_collection(CameraModel.COLLECTION_NAME)
        if collection is None:
            return []
        
        try:
            cameras = list(collection.find({'status': 'active'}))
            for camera in cameras:
                camera['_id'] = str(camera['_id'])
            return cameras
        except Exception as e:
            print(f"Error retrieving active cameras: {str(e)}")
            return []
    
    @staticmethod
    def get_camera_by_id(camera_id):
        """
        Get a camera by ID
        
        Args:
            camera_id (str): Camera ID
            
        Returns:
            dict: Camera document or None
        """
        collection = get_collection(CameraModel.COLLECTION_NAME)
        if collection is None:
            return None
        
        try:
            from bson import ObjectId
            camera = collection.find_one({'_id': ObjectId(camera_id)})
            if camera:
                camera['_id'] = str(camera['_id'])
            return camera
        except Exception as e:
            print(f"Error retrieving camera: {str(e)}")
            return None
    
    @staticmethod
    def update_camera(camera_id, update_data):
        """
        Update a camera record
        
        Args:
            camera_id (str): Camera ID
            update_data (dict): Fields to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        collection = get_collection(CameraModel.COLLECTION_NAME)
        if collection is None:
            return False
        
        try:
            from bson import ObjectId
            update_data['updated_at'] = datetime.utcnow()
            result = collection.update_one(
                {'_id': ObjectId(camera_id)},
                {'$set': update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating camera: {str(e)}")
            return False
    
    @staticmethod
    def delete_camera(camera_id):
        """
        Delete a camera record
        
        Args:
            camera_id (str): Camera ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        collection = get_collection(CameraModel.COLLECTION_NAME)
        if collection is None:
            return False
        
        try:
            from bson import ObjectId
            result = collection.delete_one({'_id': ObjectId(camera_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting camera: {str(e)}")
            return False
    
    @staticmethod
    def update_camera_stream_status(camera_id, stream_id, status):
        """
        Update camera stream status
        
        Args:
            camera_id (str): Camera ID
            stream_id (str): Stream ID or None
            status (str): Stream status
            
        Returns:
            bool: True if successful, False otherwise
        """
        update_data = {
            'stream_id': stream_id,
            'updated_at': datetime.utcnow()
        }
        
        if stream_id:
            update_data['last_stream_started'] = datetime.utcnow()
        
        return CameraModel.update_camera(camera_id, update_data)


class AdminModel:
    """Model for admin user records"""
    
    COLLECTION_NAME = 'admins'
    
    @staticmethod
    def create_admin(admin_data):
        """
        Create a new admin user
        
        Args:
            admin_data (dict): Admin information (username, email, password)
            
        Returns:
            str: Inserted document ID or None
        """
        collection = get_collection(AdminModel.COLLECTION_NAME)
        if collection is None:
            return None
        
        # Check if username or email already exists
        existing = collection.find_one({
            '$or': [
                {'username': admin_data.get('username')},
                {'email': admin_data.get('email')}
            ]
        })
        if existing:
            return None
        
        admin_data['created_at'] = datetime.utcnow()
        admin_data['role'] = admin_data.get('role', 'admin')
        admin_data['is_active'] = admin_data.get('is_active', True)
        
        try:
            result = collection.insert_one(admin_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error creating admin: {str(e)}")
            return None
    
    @staticmethod
    def get_admin_by_username(username):
        """
        Get admin by username
        
        Args:
            username (str): Admin username
            
        Returns:
            dict: Admin document or None
        """
        collection = get_collection(AdminModel.COLLECTION_NAME)
        if collection is None:
            return None
        
        try:
            admin = collection.find_one({'username': username})
            if admin:
                admin['_id'] = str(admin['_id'])
            return admin
        except Exception as e:
            print(f"Error retrieving admin: {str(e)}")
            return None
    
    @staticmethod
    def get_admin_by_email(email):
        """
        Get admin by email
        
        Args:
            email (str): Admin email
            
        Returns:
            dict: Admin document or None
        """
        collection = get_collection(AdminModel.COLLECTION_NAME)
        if collection is None:
            return None
        
        try:
            admin = collection.find_one({'email': email})
            if admin:
                admin['_id'] = str(admin['_id'])
            return admin
        except Exception as e:
            print(f"Error retrieving admin: {str(e)}")
            return None
    
    @staticmethod
    def get_admin_by_id(admin_id):
        """
        Get admin by ID
        
        Args:
            admin_id (str): Admin document ID
            
        Returns:
            dict: Admin document or None
        """
        collection = get_collection(AdminModel.COLLECTION_NAME)
        if collection is None:
            return None
        
        try:
            admin = collection.find_one({'_id': ObjectId(admin_id)})
            if admin:
                admin['_id'] = str(admin['_id'])
            return admin
        except Exception as e:
            print(f"Error retrieving admin: {str(e)}")
            return None
    
    @staticmethod
    def update_admin(admin_id, update_data):
        """
        Update admin record
        
        Args:
            admin_id (str): Admin document ID
            update_data (dict): Fields to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        collection = get_collection(AdminModel.COLLECTION_NAME)
        if collection is None:
            return False
        
        update_data['updated_at'] = datetime.utcnow()
        
        try:
            result = collection.update_one(
                {'_id': ObjectId(admin_id)},
                {'$set': update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating admin: {str(e)}")
            return False

