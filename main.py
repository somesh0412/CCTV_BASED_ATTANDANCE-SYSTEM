import cv2

# Load DNN face detection model
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

# Start webcam
cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Get frame height and width
    h, w = frame.shape[:2]

    # Convert frame to blob (input for DNN)
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    # Pass blob to network
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # Draw rectangle around face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face = frame[y1:y2, x1:x2]
            if face.size != 0:
                cv2.imshow("face", face)
                key = cv2.waitKey(1)
                filename = f"faces{count}.jpg"
                if key == ord('s'):
                    cv2.imwrite(filename,face)
                    count += 1


    # Show output
    cv2.imshow("Face Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()