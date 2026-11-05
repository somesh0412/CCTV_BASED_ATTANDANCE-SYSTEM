
import os
import cv2
import pickle
import torch
import numpy as np

from pathlib import Path
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# -----------------------------
# 1. Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "yolo" / "yolov8n-face-lindevs.pt"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.pkl"

# Set your RTSP URL in an environment variable.
RTSP_URL = os.getenv("ZEBION_RTSP_URL")

if not RTSP_URL:
    raise ValueError("Set ZEBION_RTSP_URL before running the script.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initial experimental threshold; calibrate using your own students.
MATCH_THRESHOLD = 0.65

# -----------------------------
# 2. Load YOLO face detector
# -----------------------------

print("Loading YOLO model...")
yolo = YOLO(str(MODEL_PATH))

# -----------------------------
# 3. Load FaceNet
# -----------------------------

print("Loading FaceNet...")

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)

# Convert image to tensor and normalize to [-1, 1].
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# -----------------------------
# 4. Load student embeddings
# -----------------------------

print("Loading student embeddings...")

with open(EMBEDDINGS_PATH, "rb") as file:
    student_data = pickle.load(file)

known_embeddings = []
known_names = []

def extract_records(value):
    """Find embedding records inside nested dictionaries/lists."""

    if isinstance(value, dict):
        if "embedding" in value:
            yield value
        else:
            for item in value.values():
                yield from extract_records(item)

    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from extract_records(item)

for student_name, records in student_data.items():

    for record in extract_records(records):

        embedding = np.asarray(
            record["embedding"],
            dtype=np.float32
        ).flatten()

        if embedding.shape == (512,):
            # Normalize for cosine similarity.
            norm = np.linalg.norm(embedding)

            if norm > 0:
                embedding = embedding / norm

                known_embeddings.append(embedding)
                known_names.append(student_name)

if not known_embeddings:
    raise ValueError("No valid 512-dimensional embeddings found.")

known_embeddings = np.asarray(
    known_embeddings,
    dtype=np.float32
)

print("Students loaded:", len(set(known_names)))
print("Total embeddings:", len(known_embeddings))

# -----------------------------
# 5. Generate face embedding
# -----------------------------

def get_face_embedding(face):

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face_tensor = preprocess(face_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = facenet(face_tensor)

    embedding = embedding.cpu().numpy()[0]

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    return embedding / norm

# -----------------------------
# 6. Compare with students
# -----------------------------

def recognize_student(face):

    query_embedding = get_face_embedding(face)

    if query_embedding is None:
        return "Unknown", 0.0

    similarities = np.dot(
        known_embeddings,
        query_embedding
    )

    # Get the best score for each student.
    student_scores = {}

    for name, score in zip(known_names, similarities):

        if name not in student_scores:
            student_scores[name] = float(score)

        else:
            student_scores[name] = max(
                student_scores[name],
                float(score)
            )

    best_name = max(student_scores, key=student_scores.get)
    best_score = student_scores[best_name]

    if best_score >= MATCH_THRESHOLD:
        return best_name, best_score

    return "Unknown", best_score

# -----------------------------
# 7. Connect CCTV camera
# -----------------------------

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Cannot connect to Zebion camera.")

print("Camera connected successfully!")
print("Starting face recognition...")

# -----------------------------
# 8. Live recognition
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to receive camera frame.")
        break

    results = yolo.predict(
        frame,
        conf=0.40,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Keep coordinates within frame boundaries.
            h, w = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            name, score = recognize_student(face)

            color = (
                (0, 255, 0)
                if name != "Unknown"
                else (0, 0, 255)
            )

            label = f"{name} ({score:.2f})"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

    cv2.imshow("Zebion - Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
