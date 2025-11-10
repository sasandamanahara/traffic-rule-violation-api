#!/usr/bin/env python3
"""
Script to create an admin user
"""
import bcrypt
from db_models import AdminModel

def hash_password(password):
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_admin_user():
    """Create admin user with username 'admin' and password 'Admin123'"""
    username = "admin"
    password = "Admin123"
    email = "admin@trafficviolation.com"
    
    # Check if admin already exists
    existing = AdminModel.get_admin_by_username(username)
    if existing:
        print(f"❌ Admin user '{username}' already exists!")
        print(f"   Admin ID: {existing.get('_id')}")
        print(f"   Email: {existing.get('email')}")
        return False
    
    # Hash password
    hashed_password = hash_password(password)
    
    # Create admin data
    admin_data = {
        'username': username,
        'email': email,
        'password': hashed_password,
        'role': 'admin',
        'is_active': True
    }
    
    # Create admin
    admin_id = AdminModel.create_admin(admin_data)
    
    if admin_id:
        print(f"✅ Admin user created successfully!")
        print(f"   Username: {username}")
        print(f"   Email: {email}")
        print(f"   Password: {password}")
        print(f"   Admin ID: {admin_id}")
        return True
    else:
        print("❌ Failed to create admin user. Check database connection.")
        return False

if __name__ == '__main__':
    print("Creating admin user...")
    print("-" * 50)
    create_admin_user()

