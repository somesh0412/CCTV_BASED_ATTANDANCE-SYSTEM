# CCTV-Based Attendance System (v1 — FaceNet + Haar Cascade)

Real-time face recognition attendance system.
Uses FaceNet for recognition + OpenCV Haar Cascade for detection.
NO external model files needed — Haar Cascade is built into OpenCV.

## ⚠️ Python Requirement
Requires Python 3.10 or 3.11.
TensorFlow does NOT support Python 3.12+.

## Installation

```cmd
pip install -r requirements.txt
```

## Usage

Step 1 — Collect face images:
```cmd
python capture.py
```
- Enter person name, press 's' to save faces (aim for 30-50)
- Press ESC to finish

Step 2 — Generate embeddings:
```cmd
python pickle_gen.py
```

Step 3 — Launch dashboard:
```cmd
python app.py
```
Open http://127.0.0.1:5000

## Files Generated
- faces/PersonName/*.jpg  — face images
- face_data.pkl           — FaceNet embeddings
- attendance.csv          — attendance log

## Troubleshooting
- "No face detected": improve lighting, move closer to camera
- Camera not opening: change VideoCapture(0) to VideoCapture(1)
- Recognition wrong: collect more images (50+), vary angles
