# 🗄️ MongoDB Database Auto-Creation

## How It Works

MongoDB automatically creates databases and collections when you first write data to them. You don't need to manually create them!

---

## ✅ Automatic Creation Process

### **1. Database Creation**
```python
# When you connect to MongoDB, the database is created automatically
db = client['traffic_violations']  # Database created (lazy creation)
```

The database `traffic_violations` will be created when you **first insert a document**.

### **2. Collection Creation**
```python
# Collections are created when you insert the first document
violations_collection = db['violations']  # Collection created on first insert
violations_collection.insert_one({...})   # Database and collection now exist!
```

---

## 🚀 What Happens in Your App

### **On First Connection:**
```
1. App connects to MongoDB Atlas
2. Database "traffic_violations" is referenced (but not created yet)
3. Indexes are created for "violations" and "cameras" collections
4. Collections are created automatically when indexes are created
```

### **On First Video Processing:**
```
1. Video is processed
2. Violations are detected
3. ViolationModel.create_many_violations() is called
4. Data is inserted into "violations" collection
5. Database and collection are now visible in MongoDB Atlas!
```

---

## 🔍 Verify Database Creation

### **Method 1: Using Test Script**
```bash
python3 test_mongodb.py
```

Output will show:
```
✅ SUCCESS: Connected to MongoDB!
   Database: traffic_violations
   MongoDB Version: 7.x.x
   Collections: violations, cameras
```

### **Method 2: Using API**
```bash
# Get database info
curl http://localhost:5000/api/database/info

# Response:
{
  "success": true,
  "database": {
    "database_name": "traffic_violations",
    "collections": ["violations", "cameras"],
    "server_info": "7.x.x"
  }
}
```

### **Method 3: MongoDB Atlas Dashboard**
1. Login to MongoDB Atlas
2. Go to "Database" → "Browse Collections"
3. You'll see `traffic_violations` database
4. Inside: `violations` and `cameras` collections

---

## 📊 Database Schema

### **Database: `traffic_violations`**

#### **Collection: `violations`**
```javascript
{
  _id: ObjectId,
  type: "Helmet | Triple Riding | Number Plate",
  confidence: 0.95,
  frame: 120,
  timestamp: 4.5,
  bbox: [x1, y1, x2, y2],
  snapshot_url: "http://...",
  video_file: "uuid",
  status: "pending",
  location: "Location name",
  description: "Description",
  created_at: ISODate,
  updated_at: ISODate
}
```

**Indexes:**
- `type` - For filtering by violation type
- `status` - For filtering by status
- `created_at` - For date range queries
- `created_at (descending)` - For recent violations

#### **Collection: `cameras`**
```javascript
{
  _id: ObjectId,
  name: "Camera name",
  location: "Location",
  latitude: 6.9271,
  longitude: 79.8612,
  status: "active",
  resolution: "1920x1080",
  fps: 30,
  created_at: ISODate
}
```

**Indexes:**
- `status` - For filtering active cameras
- `location` - For location-based queries

---

## 🎯 What's Been Implemented

### **Automatic Features:**

1. ✅ **Database Connection** - Connects to MongoDB Atlas
2. ✅ **Database Creation** - Creates `traffic_violations` database automatically
3. ✅ **Collection Creation** - Creates `violations` and `cameras` collections
4. ✅ **Index Creation** - Creates performance indexes automatically
5. ✅ **Data Insertion** - Saves violations when processing videos
6. ✅ **Health Checks** - API endpoints to verify database status

---

## 🧪 Test Database Creation

### **Step 1: Update .env file**
```env
MONGODB_URI=mongodb+srv://username:password@fyp.rnqakyu.mongodb.net/traffic_violations?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true
```

### **Step 2: Run Test Script**
```bash
python3 test_mongodb.py
```

This will:
- ✅ Connect to MongoDB
- ✅ Create database and collections
- ✅ Create indexes
- ✅ Insert sample violation
- ✅ Insert sample camera
- ✅ Retrieve data to verify

### **Step 3: Process a Video**
```bash
python3 app.py
# In another terminal:
curl -X POST http://localhost:5000/process-video -F "video=@your_video.mp4"
```

This will:
- ✅ Process the video
- ✅ Detect violations
- ✅ Save violations to MongoDB
- ✅ Return violation IDs

### **Step 4: Verify in MongoDB Atlas**
1. Login to MongoDB Atlas
2. Database → Browse Collections
3. See your `traffic_violations` database with data!

---

## 📝 Code Example: How It Works

```python
# database.py - Automatic initialization
def connect(self):
    # Connect to MongoDB
    self.client = MongoClient(mongodb_uri)
    
    # Get/create database (lazy creation)
    self.db = self.client['traffic_violations']
    
    # Initialize collections and indexes
    self._initialize_database()  # This creates collections!

def _initialize_database(self):
    # Create indexes (this also creates collections)
    violations_collection = self.db['violations']
    violations_collection.create_index('type')      # Collection created!
    violations_collection.create_index('status')
    violations_collection.create_index('created_at')
    
    cameras_collection = self.db['cameras']
    cameras_collection.create_index('status')       # Collection created!
    cameras_collection.create_index('location')
```

```python
# models.py - Inserting data
def create_violation(violation_data):
    collection = get_collection('violations')  # Gets existing collection
    result = collection.insert_one(violation_data)  # Inserts data
    return str(result.inserted_id)
```

```python
# app.py - Saving violations after video processing
violations_data = [...]  # Detected violations
saved_ids = ViolationModel.create_many_violations(violations_data)
# Database now has data!
```

---

## 🎉 Benefits of Auto-Creation

1. ✅ **No Manual Setup** - Database is created automatically
2. ✅ **No SQL Scripts** - No need to run creation scripts
3. ✅ **Flexible Schema** - Can add new fields anytime
4. ✅ **Index Optimization** - Indexes created for performance
5. ✅ **Easy Testing** - Just connect and start using
6. ✅ **Cloud Native** - Works seamlessly with MongoDB Atlas

---

## 🔧 Troubleshooting

### "Database doesn't appear in Atlas"
**Solution:** Insert some data first! Run `python3 test_mongodb.py`

### "Collections are empty"
**Solution:** Process a video to generate violations, or run the test script

### "Can't see indexes"
**Solution:** In MongoDB Atlas → Collections → Indexes tab

---

## 📚 Additional Resources

- **MongoDB Documentation**: https://docs.mongodb.com/manual/core/databases-and-collections/
- **Indexes**: https://docs.mongodb.com/manual/indexes/
- **MongoDB Atlas**: https://www.mongodb.com/docs/atlas/

---

**🎊 Your database will be created automatically on first use!**

Just update your `.env` file with credentials and run the test script or process a video.

