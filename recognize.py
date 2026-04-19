"""
recognize.py - Face Recognition Engine
========================================
Background thread that:
  1. Reads webcam frames
  2. Detects faces with OpenCV Haar Cascade (NO external model files)
  3. Recognizes with FaceNet (128-d embeddings + L2 distance)
  4. Marks attendance in attendance.csv (no duplicates per session)
  5. Exposes MJPEG frame generator for Flask

Requires Python 3.10 or 3.11 (TensorFlow constraint).
"""

import os
import cv2
import csv
import time
import pickle
import threading
import numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_facenet import FaceNet

# ── Configuration ─────────────────────────────────────────────────────────────
FACE_DATA_PKL         = "face_data.pkl"
ATTENDANCE_CSV        = "attendance.csv"
RECOGNITION_THRESHOLD = 0.90   # L2 distance; lower = stricter
SKIP_FRAMES           = 3      # run recognition every N frames
# ─────────────────────────────────────────────────────────────────────────────

# Haar Cascade — always bundled with OpenCV, no download needed
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Shared state (thread-safe) ────────────────────────────────────────────────
_lock             = threading.Lock()
_latest_frame     = None
_attendance_log   = []
_status_message   = "Initializing..."
_camera_active    = False
_recognized_today = set()
# ─────────────────────────────────────────────────────────────────────────────


def _load_known_faces():
    """Load FaceNet embeddings from pickle. Returns (names, embeddings_array)."""
    if not os.path.exists(FACE_DATA_PKL):
        return [], np.array([])
    with open(FACE_DATA_PKL, "rb") as f:
        data = pickle.load(f)
    names, embs = [], []
    for name, embeddings in data.items():
        for emb in embeddings:
            names.append(name)
            embs.append(emb)
    return names, np.array(embs)


def _ensure_csv():
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time"])


def _mark_attendance(name):
    """Write to CSV once per session per person."""
    if name in _recognized_today:
        return False
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        csv.writer(f).writerow([name, date_str, time_str])
    with _lock:
        _recognized_today.add(name)
        _attendance_log.insert(0, {"Name": name, "Date": date_str, "Time": time_str})
    print(f"[Attendance] ✅  {name}  {date_str}  {time_str}")
    return True


def _detect_faces_haar(frame):
    """
    Detect faces using Haar Cascade.
    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    detections = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
    )
    boxes = []
    if len(detections) > 0:
        for (x, y, w, h) in detections:
            boxes.append((x, y, x + w, y + h))   # convert to x1,y1,x2,y2
    return boxes


def _draw_overlay(frame, boxes, labels):
    """Draw bounding boxes and name labels on frame."""
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        is_known = label not in ("Unknown", "?")
        color = (0, 200, 80) if is_known else (0, 140, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_y = y1 - 12 if y1 > 30 else y2 + 24
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame,
                      (x1, label_y - th - 8), (x1 + tw + 10, label_y + 4),
                      color, -1)
        cv2.putText(frame, label, (x1 + 5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Faces: {len(boxes)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame


def _recognition_loop():
    global _latest_frame, _status_message, _camera_active

    with _lock:
        _status_message = "Loading FaceNet model..."

    # ── Load FaceNet ─────────────────────────────────────────────────────────
    try:
        embedder = FaceNet()
    except Exception as e:
        with _lock:
            _status_message = f"FaceNet load error: {e}"
        return

    # ── Load known embeddings ─────────────────────────────────────────────────
    known_names, known_embs = _load_known_faces()
    has_known = len(known_names) > 0

    with _lock:
        _status_message = (
            f"Ready — {len(set(known_names))} person(s) registered."
            if has_known else
            "No face data found. Run pickle_gen.py first."
        )

    _ensure_csv()

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with _lock:
            _status_message = "❌ Cannot open camera (index 0)."
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with _lock:
        _camera_active = True

    frame_idx, last_boxes, last_labels = 0, [], []

    while True:
        with _lock:
            if not _camera_active:
                break

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_idx += 1

        # ── Every SKIP_FRAMES: detect + recognize ─────────────────────────
        if frame_idx % SKIP_FRAMES == 0:
            boxes  = _detect_faces_haar(frame)
            labels = []

            if boxes and has_known:
                for (x1, y1, x2, y2) in boxes:
                    # Crop + resize to FaceNet input
                    face_bgr = frame[y1:y2, x1:x2]
                    if face_bgr.size == 0:
                        labels.append("Unknown")
                        continue

                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    face_rgb = cv2.resize(face_rgb, (160, 160))
                    face_batch = np.expand_dims(face_rgb, axis=0)

                    emb = embedder.embeddings(face_batch)[0]
                    dists = np.linalg.norm(known_embs - emb, axis=1)
                    min_dist = np.min(dists)
                    best_idx = int(np.argmin(dists))

                    if min_dist < RECOGNITION_THRESHOLD:
                        name = known_names[best_idx]
                        _mark_attendance(name)
                        labels.append(name)
                    else:
                        labels.append("Unknown")

            elif boxes:
                labels = ["?" for _ in boxes]

            last_boxes, last_labels = boxes, labels

        # ── Annotate + encode every frame ─────────────────────────────────
        annotated = _draw_overlay(frame.copy(), last_boxes, last_labels)
        _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _lock:
            _latest_frame = jpeg.tobytes()

    cap.release()
    with _lock:
        _camera_active = False
        _status_message = "Camera stopped."


# ── Public API used by app.py ─────────────────────────────────────────────────

def start():
    threading.Thread(target=_recognition_loop, daemon=True).start()

def stop():
    global _camera_active
    with _lock:
        _camera_active = False

def generate_frames():
    while True:
        with _lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_status():
    with _lock: return _status_message

def get_attendance():
    with _lock: return list(_attendance_log)

def get_camera_active():
    with _lock: return _camera_active
