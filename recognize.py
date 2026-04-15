import cv2
import numpy as np
import pickle
import csv
import os
#to hide tensor flow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
from keras_facenet import FaceNet


# Load FaceNet
embedder = FaceNet()

# Load saved embeddings
with open("face_data.pkl", "rb") as f:
    data = pickle.load(f)

# Attendance file
attendance_file = "attendance.csv"

# Create file if not exists
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

# Function to mark attendance
def mark_attendance(name):
    with open(attendance_file, "r+") as f:
        lines = f.readlines()
        names = [line.split(",")[0] for line in lines]

        if name not in names:
            now = datetime.now()
            time_string = now.strftime("%H:%M:%S")

            f.write(f"\n{name},{time_string}")
            print(f"{name} marked present")



# Load saved embeddings
with open("face_data.pkl", "rb") as f:
    data = pickle.load(f)

# Separate names and embeddings
known_embeddings = []
known_names = []

for name, embeddings in data.items():
    for emb in embeddings:
        known_embeddings.append(emb)
        known_names.append(name)

known_embeddings = np.array(known_embeddings)

# Start camera
cap = cv2.VideoCapture(0)

print("Recognition started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using FaceNet MTCNN
    faces = embedder.extract(rgb, threshold=0.95)

    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)

        # Crop face
        face_img = rgb[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        # Resize to 160x160
        face_img = cv2.resize(face_img, (160, 160))

        # Get embedding
        face_img = np.expand_dims(face_img, axis=0)
        emb = embedder.embeddings(face_img)[0]

        # Compare with known embeddings
        distances = np.linalg.norm(known_embeddings - emb, axis=1)
        min_dist = np.min(distances)
        index = np.argmin(distances)

        # Threshold for recognition
        if min_dist < 0.9:
            name = known_names[index]
            mark_attendance(name)
        else:
            name = "Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put name
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()