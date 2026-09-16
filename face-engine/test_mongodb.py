import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env
load_dotenv()

# Get MongoDB URI
mongo_uri = os.getenv("MONGO_URI")

if not mongo_uri:
    print("❌ MONGO_URI not found in .env")
    exit()

try:
    client = MongoClient(mongo_uri)

    # Test connection
    client.admin.command("ping")

    print("✅ MongoDB connection successful!")

except Exception as error:
    print("❌ MongoDB connection failed:")
    print(error)