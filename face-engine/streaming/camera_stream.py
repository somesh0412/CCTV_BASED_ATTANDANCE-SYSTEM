"""
streaming/camera_stream.py

Module 16B - AI Processed Live Camera Stream

Complete flow:

    Laptop Webcam
          ↓
    AttendancePipeline
          ↓
    FacePipeline
          ↓
    YOLO
          ↓
    DeepSORT
          ↓
    FaceNet
          ↓
    Matcher
          ↓
    Identity Cache
          ↓
    AttendanceEngine
          ↓
    Node.js Attendance API
          ↓
    MongoDB Atlas

    At the same time:

    Processed Frame
          ↓
    Flask
          ↓
    /video_feed
          ↓
    Teacher Dashboard
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from flask import Flask, Response, jsonify, request
from flask_cors import CORS


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# ATTENDANCE PIPELINE
# =========================================================

from pipeline.attendance_pipeline import (
    AttendancePipeline,
    draw_results,
)
from recognition.embedding import FaceEmbedder


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# GLOBAL OBJECTS
# =========================================================

camera = None
pipeline = None
registration_detector = None
registration_embedder = None


# =========================================================
# INITIALIZE PIPELINE
# =========================================================

def initialize_pipeline():

    global pipeline, registration_detector, registration_embedder

    print("========================================")
    print("Module 16B - AI Attendance Live Stream")
    print("========================================")

    print("\nLoading Attendance Pipeline...")

    pipeline = AttendancePipeline()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    registration_detector = MTCNN(keep_all=True, device=device)
    registration_embedder = FaceEmbedder(device=device)

    print("\nAttendance Pipeline ready.")


@app.post("/register/embedding")
def register_embedding():
    if registration_detector is None or registration_embedder is None:
        return jsonify({"error": "Face engine is still starting"}), 503

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")
    if not isinstance(image_data, str) or "," not in image_data:
        return jsonify({"error": "A base64 image is required"}), 400

    try:
        image_bytes = base64.b64decode(image_data.split(",", 1)[1], validate=True)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image could not be decoded")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probabilities = registration_detector.detect(rgb_image)
        if boxes is None or probabilities is None or len(boxes) == 0:
            return jsonify({"error": "No face detected. Move closer and try again."}), 422

        best_index = int(np.argmax(probabilities))
        x1, y1, x2, y2 = boxes[best_index].astype(int)
        height, width = image.shape[:2]
        padding_x = int((x2 - x1) * 0.2)
        padding_y = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(width, x2 + padding_x)
        y2 = min(height, y2 + padding_y)
        face_crop = image[y1:y2, x1:x2]

        embedding = registration_embedder.get_embedding(face_crop)
        if embedding is None or embedding.shape != (512,):
            return jsonify({"error": "Could not create a valid face embedding"}), 422

        return jsonify({"embedding": embedding.astype(float).tolist()})
    except (ValueError, TypeError, base64.binascii.Error):
        return jsonify({"error": "Invalid image data"}), 400


# =========================================================
# INITIALIZE CAMERA
# =========================================================

def initialize_camera():

    global camera

    print("\nOpening laptop webcam...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open laptop webcam."
        )

    print("Webcam opened successfully.")


# =========================================================
# FRAME GENERATOR
# =========================================================

def generate_frames():

    global camera
    global pipeline

    while True:

        # -------------------------------------------------
        # Read webcam frame
        # -------------------------------------------------

        success, frame = camera.read()

        if not success:

            print(
                "[Streaming] Failed to read webcam frame."
            )

            break

        # -------------------------------------------------
        # Run complete attendance pipeline
        # -------------------------------------------------

        try:

            (
                face_results,
                new_events,
            ) = pipeline.process_frame(frame)

        except Exception as error:

            print(
                f"[Streaming] Pipeline error: {error}"
            )

            continue

        # -------------------------------------------------
        # Print newly created attendance events
        # -------------------------------------------------

        for event in new_events:

            print("\n========================================")
            print("NEW ATTENDANCE EVENT")
            print("========================================")

            print(
                f"Student ID : {event.student_id}"
            )

            print(
                f"Name       : {event.name}"
            )

            print(
                f"Date       : {event.date}"
            )

            print(
                f"Time       : {event.time}"
            )

            print(
                f"Status     : {event.status}"
            )

            print(
                f"Track ID   : {event.track_id}"
            )

            print(
                f"Similarity : {event.similarity:.4f}"
            )

            print("========================================\n")

        # -------------------------------------------------
        # Draw recognition + attendance results
        # -------------------------------------------------

        display_frame = draw_results(
            frame,
            face_results,
            new_events,
            pipeline.attendance_engine,
        )

        # -------------------------------------------------
        # Additional streaming information
        # -------------------------------------------------

        cv2.putText(
            display_frame,
            "AI ATTENDANCE LIVE FEED",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        # -------------------------------------------------
        # Encode frame as JPEG
        # -------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            display_frame,
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        # -------------------------------------------------
        # Send frame to browser
        # -------------------------------------------------

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# =========================================================
# VIDEO STREAM ENDPOINT
# =========================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return {
        "service": "face-engine-stream",
        "status": "ok",
    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>
            CCTV AI Attendance
        </title>

    </head>

    <body>

        <h1>
            CCTV AI Attendance Live Feed
        </h1>

        <p>
            YOLO + DeepSORT + FaceNet + Attendance
        </p>

        <img
            src="/video_feed"
            width="900"
        >

    </body>

    </html>
    """


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    try:

        # -------------------------------------------------
        # Initialize AI attendance pipeline
        # -------------------------------------------------

        initialize_pipeline()

        # -------------------------------------------------
        # Initialize webcam
        # -------------------------------------------------

        initialize_camera()

        print("\n========================================")
        print("AI Attendance Streaming Server Ready")
        print("========================================")

        print(
            "\nHealth:"
        )

        print(
            "http://127.0.0.1:8000/health"
        )

        print(
            "\nAI Video:"
        )

        print(
            "http://127.0.0.1:8000/video_feed"
        )

        print(
            "\nTeacher Dashboard:"
        )

        print(
            "http://localhost:5173"
        )

        print(
            "\nPress CTRL+C to stop."
        )

        print(
            "========================================\n"
        )

        # -------------------------------------------------
        # Start Flask
        # -------------------------------------------------

        app.run(
            host="0.0.0.0",
            port=8000,
            debug=False,
            threaded=True,
        )

    finally:

        if camera is not None:

            camera.release()

        print(
            "\nWebcam released."
        )