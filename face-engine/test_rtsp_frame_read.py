import os
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

camera_url = os.getenv("ATTENDANCE_CAMERA_URL")
if not camera_url or not camera_url.strip():
    raise RuntimeError(
        "ATTENDANCE_CAMERA_URL is not configured in face-engine/.env."
    )

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

camera = None
successful_reads = 0
failed_reads = 0
start_time = time.perf_counter()

try:
    camera = cv2.VideoCapture(camera_url.strip(), cv2.CAP_FFMPEG)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        raise RuntimeError("Could not open the configured RTSP camera.")

    for frame_number in range(1, 31):
        success, frame = camera.read()
        elapsed_seconds = time.perf_counter() - start_time

        if success and frame is not None:
            successful_reads += 1
            height, width = frame.shape[:2]
            print(
                f"Frame {frame_number}: "
                f"{width}x{height}, elapsed {elapsed_seconds:.3f}s"
            )
        else:
            failed_reads += 1
            print(
                f"Frame {frame_number}: read failed, "
                f"elapsed {elapsed_seconds:.3f}s"
            )
finally:
    if camera is not None:
        camera.release()

print(f"Successful reads: {successful_reads}")
print(f"Failed reads: {failed_reads}")
