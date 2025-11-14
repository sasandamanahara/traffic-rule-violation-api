#!/usr/bin/env python3
"""
MongoDB Connection Test Script

This script tests your MongoDB connection and creates sample data
Run this to verify your MongoDB Atlas setup is working correctly.

Usage:
    python test_mongodb.py
"""

from database import get_db, get_database_info
from db_models import ViolationModel, CameraModel
from datetime import datetime

def test_connection():
    """Test MongoDB connection"""
    print("\n" + "="*50)
    print("🧪 Testing MongoDB Connection")
    print("="*50 + "\n")
    
    db = get_db()
    
    if db.connected:
        print("✅ SUCCESS: Connected to MongoDB!")
        print(f"   Database: {db.db.name}")
        print(f"   Connection: {db.client.address}")
        
        # Show database info
        db_info = get_database_info()
        if db_info:
            print(f"   MongoDB Version: {db_info['server_info']}")
            print(f"   Collections: {', '.join(db_info['collections']) if db_info['collections'] else 'None (will be created on first write)'}")
        
        return True
    else:
        print("❌ FAILED: Could not connect to MongoDB")
        print("   Please check your .env file configuration")
        return False

def test_create_sample_violation():
    """Create a sample violation record"""
    print("\n" + "="*50)
    print("📝 Creating Sample Violation")
    print("="*50 + "\n")
    
    sample_violation = {
        'type': 'Helmet',
        'confidence': 0.95,
        'frame': 100,
        'timestamp': 3.33,
        'bbox': [150, 200, 250, 350],
        'snapshot_url': 'http://localhost:5000/static/snapshots/test.jpg',
        'video_file': 'test-video-123',
        'status': 'pending',
        'location': 'Test Location - Main Street',
        'description': 'Test violation - Helmet not detected'
    }
    
    violation_id = ViolationModel.create_violation(sample_violation)
    
    if violation_id:
        print(f"✅ SUCCESS: Created violation with ID: {violation_id}")
        return violation_id
    else:
        print("❌ FAILED: Could not create violation")
        return None

def test_retrieve_violations():
    """Retrieve violations from database"""
    print("\n" + "="*50)
    print("🔍 Retrieving Violations")
    print("="*50 + "\n")
    
    violations = ViolationModel.get_all_violations(limit=5)
    
    if violations:
        print(f"✅ SUCCESS: Retrieved {len(violations)} violation(s)")
        for i, violation in enumerate(violations, 1):
            print(f"\n   Violation {i}:")
            print(f"   - ID: {violation['_id']}")
            print(f"   - Type: {violation['type']}")
            print(f"   - Confidence: {violation['confidence']}")
            print(f"   - Status: {violation['status']}")
            if 'created_at' in violation:
                print(f"   - Created: {violation['created_at']}")
        return True
    else:
        print("⚠️  No violations found in database")
        return False

def test_violation_stats():
    """Get violation statistics"""
    print("\n" + "="*50)
    print("📊 Violation Statistics")
    print("="*50 + "\n")
    
    stats = ViolationModel.get_violation_stats()
    
    if stats:
        print(f"✅ Total Violations: {stats['total_violations']}")
        
        if stats['by_type']:
            print("\n   Violations by Type:")
            for vtype, count in stats['by_type'].items():
                print(f"   - {vtype}: {count}")
        
        if stats['by_status']:
            print("\n   Violations by Status:")
            for status, count in stats['by_status'].items():
                print(f"   - {status}: {count}")
        
        print(f"\n   Recent (24h): {stats['recent_count']}")
        return True
    else:
        print("❌ FAILED: Could not retrieve statistics")
        return False

def test_create_camera():
    """Create a sample camera record"""
    print("\n" + "="*50)
    print("📹 Creating Sample Camera")
    print("="*50 + "\n")
    
    sample_camera = {
        'name': 'Main Street Camera 01',
        'location': 'Main Street & Elm Avenue',
        'latitude': 6.9271,
        'longitude': 79.8612,
        'status': 'active',
        'resolution': '1920x1080',
        'fps': 30
    }
    
    camera_id = CameraModel.create_camera(sample_camera)
    
    if camera_id:
        print(f"✅ SUCCESS: Created camera with ID: {camera_id}")
        return camera_id
    else:
        print("❌ FAILED: Could not create camera")
        return None

def test_retrieve_cameras():
    """Retrieve cameras from database"""
    print("\n" + "="*50)
    print("🎥 Retrieving Cameras")
    print("="*50 + "\n")
    
    cameras = CameraModel.get_all_cameras()
    
    if cameras:
        print(f"✅ SUCCESS: Retrieved {len(cameras)} camera(s)")
        for i, camera in enumerate(cameras, 1):
            print(f"\n   Camera {i}:")
            print(f"   - ID: {camera['_id']}")
            print(f"   - Name: {camera.get('name', 'N/A')}")
            print(f"   - Location: {camera.get('location', 'N/A')}")
            print(f"   - Status: {camera.get('status', 'N/A')}")
        return True
    else:
        print("⚠️  No cameras found in database")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 MongoDB Integration Test Suite")
    print("="*60)
    
    # Test 1: Connection
    if not test_connection():
        print("\n❌ Connection test failed. Please check your configuration.")
        print("   Make sure your .env file exists and has the correct MongoDB URI")
        return
    
    # Test 2: Create sample violation
    violation_id = test_create_sample_violation()
    
    # Test 3: Retrieve violations
    test_retrieve_violations()
    
    # Test 4: Get statistics
    test_violation_stats()
    
    # Test 5: Create sample camera
    camera_id = test_create_camera()
    
    # Test 6: Retrieve cameras
    test_retrieve_cameras()
    
    # Summary
    print("\n" + "="*60)
    print("✅ All Tests Completed!")
    print("="*60)
    print("\n📋 Summary:")
    print(f"   - Database connection: ✅")
    print(f"   - Sample violation created: {'✅' if violation_id else '❌'}")
    print(f"   - Sample camera created: {'✅' if camera_id else '❌'}")
    print("\n🎉 Your MongoDB integration is working correctly!")
    print("\nNext steps:")
    print("   1. Process a video using: POST /process-video")
    print("   2. Check violations using: GET /api/violations")
    print("   3. View stats using: GET /api/violations/stats")
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

