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

import hmac
import logging
import os
import sys
import uuid
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

LOGGER = logging.getLogger(__name__)


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# =========================================================
# ATTENDANCE PIPELINE
# =========================================================

from pipeline.attendance_pipeline import (
    AttendancePipeline,
    draw_results,
)


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)
CORS(app)

ENROLLMENT_API_SECRET_ENV = "FACE_ENROLLMENT_API_SECRET"
ENROLLMENT_API_SECRET_HEADER = "X-Enrollment-Secret"


# =========================================================
# GLOBAL OBJECTS
# =========================================================

camera = None
pipeline = None
camera_lock = RLock()


# =========================================================
# FACE ENROLLMENT GLOBAL OBJECTS
# =========================================================

enrollment_detector = None
enrollment_embedder = None

enrollment_sessions = {}
enrollment_lock = RLock()
enrollment_model_lock = RLock()
enrollment_embedding_delivered = set()

# =========================================================
# INITIALIZE PIPELINE
# =========================================================

def initialize_pipeline():

    global pipeline

    print("========================================")
    print("Module 16B - AI Attendance Live Stream")
    print("========================================")

    print("\nLoading Attendance Pipeline...")

    pipeline = AttendancePipeline()

    print("\nAttendance Pipeline ready.")
# =========================================================
# INITIALIZE FACE ENROLLMENT MODELS
# =========================================================

def initialize_enrollment_models():

    global enrollment_detector
    global enrollment_embedder

    print("\nLoading Face Enrollment Models...")

    from detection.yolo_detector import FaceDetector
    from recognition.embedding import FaceEmbedder

    enrollment_detector = FaceDetector(device="cpu")
    enrollment_embedder = FaceEmbedder(device="cpu")

    print("Face Enrollment Models loaded successfully.")


def _enrollment_secret_is_valid():
    """Validate the shared secret for an enrollment API request."""
    configured_secret = os.getenv(ENROLLMENT_API_SECRET_ENV)
    supplied_secret = request.headers.get(ENROLLMENT_API_SECRET_HEADER)

    return bool(
        configured_secret
        and supplied_secret
        and hmac.compare_digest(supplied_secret, configured_secret)
    )


def _enrollment_authentication_error():
    """Return the standard response for missing or invalid API secrets."""
    if _enrollment_secret_is_valid():
        return None

    return jsonify({
        "error": "Missing or invalid enrollment API secret."
    }), 401


def _parse_enrollment_session_id(session_id):
    """Validate a UUID session ID and return its canonical string form."""
    try:
        parsed_id = uuid.UUID(session_id)
    except (AttributeError, ValueError, TypeError):
        return None

    if str(parsed_id) != session_id.lower():
        return None

    return str(parsed_id)


def _get_enrollment_manager(session_id):
    """Read an enrollment manager without holding the lock during inference."""
    with enrollment_lock:
        return enrollment_sessions.get(session_id)


def _embedding_response(manager):
    """Return final embeddings once, after a manager reaches completion."""
    with enrollment_lock:
        if manager in enrollment_embedding_delivered:
            return None
        enrollment_embedding_delivered.add(manager)

    try:
        return [embedding.tolist() for embedding in manager.get_embeddings()]
    except RuntimeError:
        with enrollment_lock:
            enrollment_embedding_delivered.discard(manager)
        raise


def _enrollment_result_response(result, manager):
    """Build the process-frame JSON response and deliver completion embeddings."""
    progress = result["progress"]
    response = {
        "accepted": result["accepted"],
        "reason": result["reason"],
        "progress": progress,
    }

    if progress["current_status"] == "completed":
        try:
            embeddings = _embedding_response(manager)
        except RuntimeError:
            return jsonify({
                "error": "Enrollment completed but embeddings could not be retrieved."
            }), 500

        if embeddings is not None:
            response["embeddings"] = embeddings

    return jsonify(response), 200


def _refresh_enrollment_timeout(manager):
    """Apply the manager's timeout policy before returning progress."""
    with enrollment_model_lock:
        progress = manager.get_progress()
        if progress["current_status"] == "active" and manager._has_timed_out():
            manager._timeout_result()
        return manager.get_progress()

# =========================================================
# INITIALIZE CAMERA
# =========================================================

def initialize_camera():
    """Try to connect the attendance CCTV without stopping Flask startup."""

    global camera

    with camera_lock:
        if camera is not None and camera.isOpened():
            return True

        camera_url = os.getenv("ATTENDANCE_CAMERA_URL")
        if not camera_url or not camera_url.strip():
            LOGGER.warning(
                "Attendance CCTV unavailable: ATTENDANCE_CAMERA_URL is not configured."
            )
            return False

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        LOGGER.info("Opening configured attendance camera...")
        capture = cv2.VideoCapture(camera_url.strip(), cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not capture.isOpened():
            capture.release()
            LOGGER.warning("Attendance CCTV unavailable: could not open configured camera.")
            return False

        camera = capture
        LOGGER.info("Attendance camera opened successfully.")
        return True


def release_camera():
    """Release the attendance CCTV and clear the shared camera reference."""
    global camera

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None


# =========================================================
# FRAME GENERATOR
# =========================================================

def generate_frames():

    global camera
    global pipeline

    while True:

        # -------------------------------------------------
        # Read attendance CCTV frame
        # -------------------------------------------------

        with camera_lock:
            active_camera = camera

        if active_camera is None or not active_camera.isOpened():
            LOGGER.warning("Attendance stream stopped because CCTV is unavailable.")
            return

        success, frame = active_camera.read()

        if not success:

            LOGGER.warning("Attendance CCTV stopped returning frames.")
            release_camera()
            return

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
# FACE ENROLLMENT ENDPOINTS
# =========================================================

@app.route("/enrollment/start", methods=["POST"])
def enrollment_start():
    """Create and start one in-memory enrollment session."""
    authentication_error = _enrollment_authentication_error()
    if authentication_error is not None:
        return authentication_error

    if enrollment_detector is None or enrollment_embedder is None:
        return jsonify({
            "error": "Enrollment models are not initialized."
        }), 503

    session_id = str(uuid.uuid4())
    try:
        from enrollment import FaceEnrollmentManager

        manager = FaceEnrollmentManager(
            detector=enrollment_detector,
            embedder=enrollment_embedder,
        )
        progress = manager.start_session()
    except (RuntimeError, ValueError) as error:
        return jsonify({
            "error": f"Could not start enrollment: {error}"
        }), 503

    with enrollment_lock:
        enrollment_sessions[session_id] = manager
        enrollment_embedding_delivered.discard(manager)

    return jsonify({
        "session_id": session_id,
        "progress": progress,
    }), 201


@app.route("/enrollment/process-frame", methods=["POST"])
def enrollment_process_frame():
    """Decode and process one uploaded frame for an enrollment session."""
    authentication_error = _enrollment_authentication_error()
    if authentication_error is not None:
        return authentication_error

    raw_session_id = request.form.get("session_id") or request.args.get("session_id")
    session_id = _parse_enrollment_session_id(raw_session_id)
    if session_id is None:
        return jsonify({"error": "A valid UUID session_id is required."}), 400

    manager = _get_enrollment_manager(session_id)
    if manager is None:
        return jsonify({"error": "Enrollment session not found."}), 404

    uploaded_file = request.files.get("frame") or request.files.get("image")
    if uploaded_file is None:
        return jsonify({
            "error": "Upload an image using the 'frame' multipart field."
        }), 400

    image_bytes = uploaded_file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded image is empty."}), 400
    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "Uploaded image is too large."}), 413

    encoded_image = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return jsonify({"error": "Uploaded data is not a valid image."}), 400

    try:
        with enrollment_model_lock:
            result = manager.process_frame(frame)
    except (cv2.error, RuntimeError, ValueError) as error:
        return jsonify({
            "error": f"Enrollment frame processing failed: {error}"
        }), 500

    progress = result["progress"]
    if progress["current_status"] == "failed":
        return jsonify({
            "accepted": False,
            "reason": result["reason"],
            "progress": progress,
        }), 408
    if progress["current_status"] == "cancelled":
        return jsonify({
            "accepted": False,
            "reason": result["reason"],
            "progress": progress,
        }), 409

    return _enrollment_result_response(result, manager)


@app.route("/enrollment/progress/<session_id>", methods=["GET"])
def enrollment_progress(session_id):
    """Return current progress for an enrollment session."""
    authentication_error = _enrollment_authentication_error()
    if authentication_error is not None:
        return authentication_error

    normalized_session_id = _parse_enrollment_session_id(session_id)
    if normalized_session_id is None:
        return jsonify({"error": "A valid UUID session_id is required."}), 400

    manager = _get_enrollment_manager(normalized_session_id)
    if manager is None:
        return jsonify({"error": "Enrollment session not found."}), 404

    progress = _refresh_enrollment_timeout(manager)
    return jsonify({
        "session_id": normalized_session_id,
        "progress": progress,
    }), 200


@app.route("/enrollment/cancel/<session_id>", methods=["POST"])
def enrollment_cancel(session_id):
    """Cancel an active session and discard its in-memory embeddings."""
    authentication_error = _enrollment_authentication_error()
    if authentication_error is not None:
        return authentication_error

    normalized_session_id = _parse_enrollment_session_id(session_id)
    if normalized_session_id is None:
        return jsonify({"error": "A valid UUID session_id is required."}), 400

    manager = _get_enrollment_manager(normalized_session_id)
    if manager is None:
        return jsonify({"error": "Enrollment session not found."}), 404

    with enrollment_model_lock:
        progress = manager.get_progress()
        if progress["current_status"] == "completed":
            return jsonify({
                "error": "Completed enrollment sessions cannot be cancelled.",
                "progress": progress,
            }), 409
        manager.cancel()
        progress = manager.get_progress()

    with enrollment_lock:
        enrollment_embedding_delivered.discard(manager)

    return jsonify({
        "session_id": normalized_session_id,
        "progress": progress,
    }), 200


# =========================================================
# VIDEO STREAM ENDPOINT
# =========================================================

@app.route("/video_feed")
def video_feed():
    with camera_lock:
        camera_available = camera is not None and camera.isOpened()

    if not camera_available:
        return jsonify({
            "available": False,
            "error": "Attendance CCTV is unavailable. Connect it and retry camera initialization.",
        }), 503

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.route("/camera/retry", methods=["POST"])
def retry_camera():
    """Retry connecting the attendance CCTV after it becomes available."""
    if initialize_camera():
        return jsonify({
            "available": True,
            "message": "Attendance CCTV connected successfully.",
        }), 200

    return jsonify({
        "available": False,
        "error": "Attendance CCTV is still unavailable.",
    }), 503


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():
    with camera_lock:
        camera_available = camera is not None and camera.isOpened()
    return {
        "service": "face-engine-stream",
        "status": "ok",
        "attendance_camera": "connected" if camera_available else "unavailable",
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
        initialize_enrollment_models()

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
        release_camera()
        print("\nAttendance camera released.")