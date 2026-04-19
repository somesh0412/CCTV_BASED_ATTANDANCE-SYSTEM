"""
capture.py - Face Data Collection Script
==========================================
Uses OpenCV Haar Cascade — NO external model files needed.
Press 's' to save a face, ESC or 'q' to quit.
Aim for 30-50 images per person.

Usage: python capture.py
"""

import cv2
import os

FACES_DIR = "faces"

# Haar Cascade is built into OpenCV — always available, no download needed
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)   # improves detection in varied lighting
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )
    # Returns list of (x, y, w, h)
    return faces if len(faces) > 0 else []

def main():
    name = input("Enter the person's name (no spaces, e.g. JohnDoe): ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    existing = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
    count = existing

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Try changing VideoCapture(0) to VideoCapture(1).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\nCamera opened. Capturing for: {name}")
    print("Press 's' to SAVE a face  |  ESC or 'q' to QUIT\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        faces = detect_faces(frame)
        display = frame.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 220, 100), 2)

        # HUD
        status = f"Face detected!" if len(faces) > 0 else "No face — move closer"
        color  = (0, 220, 100) if len(faces) > 0 else (0, 100, 255)
        cv2.putText(display, f"{name}  |  Saved: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(display, "Press 's' to save  |  ESC to quit", (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Capture Faces — " + name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Add small padding around face crop
                pad = 20
                x1 = max(0, x - pad);  y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    path = os.path.join(person_dir, f"{name}_{count:04d}.jpg")
                    cv2.imwrite(path, face_crop)
                    count += 1
                    print(f"  Saved [{count}] → {path}")
            else:
                print("  No face detected — position your face in the frame first.")

        elif key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

    saved = count - existing
    print(f"\n✅ Done!  Saved {saved} images for '{name}' in '{person_dir}'")
    if saved > 0:
        print("→  Next step: python pickle_gen.py")
    else:
        print("⚠  No images saved. Try again with better lighting.")

if __name__ == "__main__":
    main()
