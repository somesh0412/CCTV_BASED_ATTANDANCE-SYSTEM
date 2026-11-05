from pathlib import Path
import cv2
from ultralytics import YOLO

import cv2


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "yolo" / "yolov8n-face-lindevs.pt"

print("Model exists:", MODEL_PATH.exists())
print("Model path:", MODEL_PATH)

model = YOLO(str(MODEL_PATH))

# Enter your working RTSP URL
RTSP_URL = "rtsp://admin:Somesh1234@10.61.220.166:554/onvif1"

import os
import cv2

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Cannot connect to camera")
    exit()

print("Camera connected. Starting YOLO face detection...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to read frame")
        break

    results = model.predict(frame, conf=0.40, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("Zebion - YOLO Face Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
