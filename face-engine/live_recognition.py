import os
import cv2
import numpy as np
import torch

from PIL import Image
from dotenv import load_dotenv
from pymongo import MongoClient
from facenet_pytorch import MTCNN, InceptionResnetV1


# ============================================
# 1. Configuration
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================
# 2. Load Face Detection Model
# ============================================

print("Loading face detector...")

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    device=device
)

print("Face detector loaded")


# ============================================
# 3. Load FaceNet
# ============================================

print("Loading FaceNet...")

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)

print("FaceNet loaded")


# ============================================
# 4. Connect MongoDB
# ============================================

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

if not mongo_uri:
    print("❌ MONGO_URI not found in .env")
    exit()

try:
    client = MongoClient(mongo_uri)

    client.admin.command("ping")

    db = client["test"]

    students_collection = db["students"]

    print("✅ MongoDB connected")

except Exception as error:
    print("❌ MongoDB connection failed")
    print(error)
    exit()


# ============================================
# 5. Load student embeddings from MongoDB
# ============================================

students = []

for student in students_collection.find():

    student_id = student["studentId"]
    name = student["name"]

    embeddings = []

    for stored_embedding in student["faceEmbeddings"]:

        embeddings.append(
            np.array(
                stored_embedding,
                dtype=np.float32
            )
        )

    students.append({
        "studentId": student_id,
        "name": name,
        "embeddings": embeddings
    })

print(f"✅ Loaded {len(students)} students from MongoDB")


# ============================================
# 6. Start laptop webcam
# ============================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("❌ Could not open webcam")
    client.close()
    exit()

print("✅ Webcam started")
print("Press Q to stop")


# ============================================
# 7. Live recognition
# ============================================

while True:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read webcam frame")
        break

    # Convert OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL image
    pil_image = Image.fromarray(rgb_frame)

    # ----------------------------------------
    # Detect face
    # ----------------------------------------

    face_tensor, probability = mtcnn(
        pil_image,
        return_prob=True
    )

    name = "No face"
    student_id = ""
    distance_text = ""

    if face_tensor is not None:

        # ------------------------------------
        # Generate FaceNet embedding
        # ------------------------------------

        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():

            embedding = facenet(face_tensor)

        new_embedding = (
            embedding.cpu()
            .numpy()[0]
        )

        # ------------------------------------
        # Compare with MongoDB
        # ------------------------------------

        best_student = None
        best_distance = float("inf")

        for student in students:

            for stored_embedding in student["embeddings"]:

                distance = np.linalg.norm(
                    new_embedding - stored_embedding
                )

                if distance < best_distance:

                    best_distance = distance

                    best_student = student

        # ------------------------------------
        # Display best match
        # ------------------------------------

        if best_student:

            name = best_student["name"]
            student_id = best_student["studentId"]

            distance_text = (
                f"Distance: {best_distance:.3f}"
            )

    # ========================================
    # Display result
    # ========================================

    cv2.putText(
        frame,
        f"Student: {name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if student_id:

        cv2.putText(
            frame,
            f"ID: {student_id}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    if distance_text:

        cv2.putText(
            frame,
            distance_text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.imshow(
        "CCTV Face Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================
# 8. Cleanup
# ============================================

camera.release()
cv2.destroyAllWindows()
client.close()

print("Recognition stopped")