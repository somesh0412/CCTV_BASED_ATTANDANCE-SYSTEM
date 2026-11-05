import cv2

RTSP_URL = "rtsp://admin:Somesh1234@10.61.220.166:554/onvif1"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Cannot connect to Zebion camera")
    exit()

print("Zebion camera connected successfully!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to receive frame")
        break

    cv2.imshow("Zebion CCTV Live Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()