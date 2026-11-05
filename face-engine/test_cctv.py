
import os
import cv2
from dotenv import load_dotenv

load_dotenv()

rtsp_url = os.getenv("RTSP_URL")

if not rtsp_url:
    raise ValueError("RTSP_URL not found in .env")

print("Connecting to CP PLUS CCTV...")

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Could not connect to CCTV")
    exit()

print("CCTV connected successfully!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to receive frame")
        break

    cv2.imshow("CP PLUS - Live CCTV", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
