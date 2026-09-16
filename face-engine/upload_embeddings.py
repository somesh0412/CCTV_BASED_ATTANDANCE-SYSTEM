import os
import pickle
from dotenv import load_dotenv
from pymongo import MongoClient

# -----------------------------
# 1. Load environment variables
# -----------------------------

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

if not mongo_uri:
    print("❌ MONGO_URI not found in .env")
    exit()


# -----------------------------
# 2. Connect to MongoDB
# -----------------------------

try:
    client = MongoClient(mongo_uri)

    # Test connection
    client.admin.command("ping")

    print("✅ MongoDB connection successful")

except Exception as error:
    print("❌ MongoDB connection failed")
    print(error)
    exit()


# -----------------------------
# 3. Select database
# -----------------------------

db = client["test"]

students_collection = db["students"]


# -----------------------------
# 4. Load embeddings.pkl
# -----------------------------

with open("embeddings.pkl", "rb") as file:
    embeddings_data = pickle.load(file)

print(f"📂 Students found: {len(embeddings_data)}")


# -----------------------------
# 5. Upload students
# -----------------------------

student_number = 1

for student_name, student_embeddings in embeddings_data.items():

    student_id = f"STU{student_number:03d}"

    document = {
        "studentId": student_id,
        "name": student_name,
        "faceRegistered": True,
        "faceEmbeddings": [
            item["embedding"]
            for item in student_embeddings
        ]
    }

    # Prevent duplicate student IDs
    students_collection.update_one(
        {"studentId": student_id},
        {"$set": document},
        upsert=True
    )

    print(
        f"✅ {student_id} → {student_name} "
        f"({len(student_embeddings)} embeddings)"
    )

    student_number += 1


# -----------------------------
# 6. Finish
# -----------------------------

print("\n===================================")
print("Embedding upload completed")
print("===================================")

print(
    f"Total students uploaded: "
    f"{len(embeddings_data)}"
)

client.close()