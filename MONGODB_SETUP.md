# MongoDB Integration Setup Guide

This guide will help you set up and use MongoDB with your Traffic Rule Violation Detection API.

## 📋 Table of Contents
1. [Create MongoDB Atlas Account](#create-mongodb-atlas-account)
2. [Configure Your Project](#configure-your-project)
3. [Install Dependencies](#install-dependencies)
4. [Run the Application](#run-the-application)
5. [API Endpoints](#api-endpoints)
6. [Testing the Integration](#testing-the-integration)

---

## 🚀 Create MongoDB Atlas Account

### Step 1: Sign Up
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Sign up using:
   - Email address
   - Google account
   - GitHub account

### Step 2: Create a Free Cluster
1. After login, click **"Build a Database"**
2. Select **"M0 FREE"** tier (512MB storage, perfect for development)
3. Choose a cloud provider (AWS, Google Cloud, or Azure)
4. Select a region closest to your location
5. Name your cluster (e.g., `TrafficViolationCluster`)
6. Click **"Create Cluster"** (takes 3-5 minutes)

### Step 3: Create Database User
1. Click **"Database Access"** in the left sidebar
2. Click **"Add New Database User"**
3. Choose **"Password"** authentication method
4. Username: `trafficadmin` (or your choice)
5. Password: Create a strong password (SAVE THIS!)
6. Database User Privileges: **"Read and write to any database"**
7. Click **"Add User"**

### Step 4: Configure Network Access
1. Click **"Network Access"** in the left sidebar
2. Click **"Add IP Address"**
3. For development: Click **"Allow Access from Anywhere"** (0.0.0.0/0)
   - ⚠️ **Security Note**: For production, restrict to specific IPs
4. Click **"Confirm"**

### Step 5: Get Connection String
1. Go to **"Database"** section
2. Click **"Connect"** button on your cluster
3. Choose **"Connect your application"**
4. Driver: **Python** | Version: **3.12 or later**
5. Copy the connection string (looks like this):
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

---

## ⚙️ Configure Your Project

### Update Environment Variables

1. Navigate to the API directory:
   ```bash
   cd /Users/sasandamanahara/Documents/FYP/traffic-rule-violation-api
   ```

2. Open the `.env.example` file and copy it to `.env`:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file and update with your MongoDB credentials:
   ```env
   # MongoDB Configuration
   MONGODB_URI=mongodb+srv://trafficadmin:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/traffic_violations?retryWrites=true&w=majority
   
   # Database Name
   DATABASE_NAME=traffic_violations
   
   # Flask Configuration
   FLASK_ENV=development
   FLASK_DEBUG=True
   
   # API Configuration
   API_PORT=5000
   ```

   **Replace:**
   - `trafficadmin` with your database username
   - `YOUR_PASSWORD` with your database password
   - `cluster0.xxxxx` with your actual cluster address

---

## 📦 Install Dependencies

Install the new MongoDB dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pymongo[srv] python-dotenv dnspython
```

---

## 🏃 Run the Application

Start the Flask API:

```bash
python app.py
```

You should see:
```
✅ Successfully connected to MongoDB database: traffic_violations
 * Running on http://127.0.0.1:5000
```

If MongoDB is not configured, you'll see:
```
⚠️  WARNING: MongoDB URI not configured. Using local storage fallback.
```

---

## 🔌 API Endpoints

### Health Check
```bash
GET /api/health
```
Check API and database connection status.

**Response:**
```json
{
  "api_status": "running",
  "database_connected": true,
  "database_type": "MongoDB"
}
```

### Get All Violations
```bash
GET /api/violations?limit=50&skip=0&type=Helmet&status=pending
```

**Query Parameters:**
- `limit` (optional): Number of records to return (default: 100)
- `skip` (optional): Number of records to skip (default: 0)
- `type` (optional): Filter by violation type (e.g., "Helmet", "Triple Riding")
- `status` (optional): Filter by status (e.g., "pending", "reviewed")

**Response:**
```json
{
  "success": true,
  "count": 15,
  "violations": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "type": "Helmet",
      "confidence": 0.95,
      "frame": 120,
      "timestamp": 4.5,
      "bbox": [100, 150, 200, 250],
      "snapshot_url": "http://localhost:5000/static/snapshots/abc.jpg",
      "status": "pending",
      "created_at": "2024-11-10T10:30:00Z"
    }
  ]
}
```

### Get Single Violation
```bash
GET /api/violations/<violation_id>
```

### Update Violation
```bash
PUT /api/violations/<violation_id>
Content-Type: application/json

{
  "status": "reviewed",
  "notes": "Violation confirmed"
}
```

### Delete Violation
```bash
DELETE /api/violations/<violation_id>
```

### Get Violation Statistics
```bash
GET /api/violations/stats
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_violations": 234,
    "by_type": {
      "Helmet": 120,
      "Triple Riding": 114
    },
    "by_status": {
      "pending": 180,
      "reviewed": 54
    },
    "recent_count": 45
  }
}
```

### Camera Management
```bash
GET /api/cameras              # Get all cameras
POST /api/cameras             # Create new camera
```

---

## 🧪 Testing the Integration

### Test 1: Health Check
```bash
curl http://localhost:5000/api/health
```

### Test 2: Process a Video (Creates Violations)
```bash
curl -X POST http://localhost:5000/process-video \
  -F "video=@path/to/your/video.mp4"
```

### Test 3: Retrieve Violations from MongoDB
```bash
curl http://localhost:5000/api/violations?limit=10
```

### Test 4: Get Statistics
```bash
curl http://localhost:5000/api/violations/stats
```

---

## 📊 Data Structure

### Violation Document Schema
```json
{
  "_id": "ObjectId",
  "type": "Helmet | Triple Riding | Number Plate",
  "confidence": 0.95,
  "frame": 120,
  "timestamp": 4.5,
  "bbox": [x1, y1, x2, y2],
  "snapshot_url": "URL to snapshot image",
  "video_file": "UUID of source video",
  "status": "pending | reviewed | approved | rejected",
  "location": "Camera location",
  "description": "Violation description",
  "created_at": "ISO datetime",
  "updated_at": "ISO datetime"
}
```

---

## 🔍 View Your Data in MongoDB Atlas

1. Go to [MongoDB Atlas](https://cloud.mongodb.com)
2. Click on **"Browse Collections"**
3. Select your database: `traffic_violations`
4. You'll see collections:
   - `violations` - All detected violations
   - `cameras` - Camera information

---

## 🛠️ Troubleshooting

### Issue: "Failed to connect to MongoDB"
**Solution:**
1. Check your `.env` file has the correct MongoDB URI
2. Verify network access allows your IP (0.0.0.0/0 for development)
3. Ensure database user credentials are correct

### Issue: "DNS resolution failed"
**Solution:**
```bash
pip install dnspython
```

### Issue: App works but doesn't save to MongoDB
**Solution:**
- Check console output for connection messages
- The app will continue to work without MongoDB (local fallback)
- Verify your `.env` file exists and has correct settings

---

## 🎯 Next Steps

1. ✅ Set up MongoDB Atlas account
2. ✅ Configure `.env` file with your connection string
3. ✅ Install dependencies
4. ✅ Run the application
5. ✅ Test video processing and violation storage
6. 🔄 Integrate frontend to display MongoDB data
7. 📊 Add analytics and reporting features

---

## 💡 Tips

- **Free Tier Limits**: 512MB storage, perfect for development
- **Backup**: MongoDB Atlas provides automatic backups
- **Monitoring**: Use Atlas monitoring to track database performance
- **Security**: Always use environment variables for credentials
- **Production**: When deploying, update network access rules to specific IPs

---

## 📞 Support

If you encounter issues:
1. Check the console output for error messages
2. Verify MongoDB Atlas cluster is running
3. Test connection using MongoDB Compass
4. Review the `.env` configuration

---

**🎉 Your MongoDB integration is now complete!**

The system will automatically save all detected violations to MongoDB when processing videos.

