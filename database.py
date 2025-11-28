"""
Database configuration and connection management for MongoDB
"""
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Database:
    """MongoDB database connection manager"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
        
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            mongodb_uri = "mongodb+srv://nipunikumudika:nipunikumudika@cluster0.txmga7v.mongodb.net/fyp?retryWrites=true&w=majority"
            database_name = os.getenv('DATABASE_NAME', 'fyp')
            
            if not mongodb_uri or mongodb_uri == 'mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/traffic_violations?retryWrites=true&w=majority':
                print("⚠️  WARNING: MongoDB URI not configured. Using local storage fallback.")
                print("   Please update your .env file with your MongoDB Atlas connection string.")
                self.connected = False
                return False
            
            # Create MongoDB client
            self.client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database (creates automatically if doesn't exist)
            self.db = self.client[database_name]
            self.connected = True
            
            # Initialize database collections and indexes
            self._initialize_database()
            
            print(f"Successfully connected to MongoDB database: {database_name}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"Failed to connect to MongoDB: {str(e)}")
            print("   Using local storage fallback.")
            self.connected = False
            return False
        except Exception as e:
            print(f"Error connecting to MongoDB: {str(e)}")
            self.connected = False
            return False
    
    def get_collection(self, collection_name):
        """Get a specific collection from the database"""
        if not self.connected or self.db is None:
            return None
        return self.db[collection_name]
    
    def _initialize_database(self):
        """Initialize database with collections and indexes"""
        try:
            # Create collections if they don't exist (MongoDB creates them automatically on first insert)
            # But we'll create indexes for better performance
            
            # Violations collection indexes
            violations_collection = self.db['violations']
            violations_collection.create_index('type')
            violations_collection.create_index('status')
            violations_collection.create_index('created_at')
            violations_collection.create_index([('created_at', -1)])  # Descending for recent queries
            
            # Cameras collection indexes
            cameras_collection = self.db['cameras']
            cameras_collection.create_index('status')
            cameras_collection.create_index('location')
            
            # Admins collection indexes
            admins_collection = self.db['admins']
            admins_collection.create_index('username', unique=True)
            admins_collection.create_index('email', unique=True)
            admins_collection.create_index('is_active')
            
            print(f"   Database '{self.db.name}' initialized with collections and indexes")
            
        except Exception as e:
            print(f"   Warning: Could not create indexes: {str(e)}")
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.connected = False
            print("🔌 MongoDB connection closed")

# Global database instance
db_instance = Database()

def get_db():
    """Get the database instance"""
    if not db_instance.connected:
        db_instance.connect()
    return db_instance

def get_collection(collection_name):
    """Get a collection from the database"""
    db = get_db()
    if db.connected:
        return db.get_collection(collection_name)
    return None

def get_database_info():
    """Get information about the database"""
    db = get_db()
    if not db.connected:
        return None
    
    try:
        info = {
            'database_name': db.db.name,
            'collections': db.db.list_collection_names(),
            'server_info': db.client.server_info()['version']
        }
        return info
    except Exception as e:
        print(f"Error getting database info: {str(e)}")
        return None

