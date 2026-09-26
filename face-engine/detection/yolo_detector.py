from dataclasses import dataclass
import os
import cv2

from ultralytics import YOLO


# Path to our face-specific YOLO model
MODEL_PATH = "models/yolo/yolov8n-face-lindevs.pt"


@dataclass
class Detection:
    """
    Represents one detected face.
    """

    bbox: tuple
    confidence: float

    @property
    def width(self):
        x1, y1, x2, y2 = self.bbox
        return x2 - x1

    @property
    def height(self):
        x1, y1, x2, y2 = self.bbox
        return y2 - y1


class FaceDetector:
    """
    YOLO-based face detector.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf_threshold=0.5,
        imgsz=640,
        device="cpu"
    ):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"\nYOLO face model not found:\n"
                f"{model_path}\n\n"
                f"Expected file:\n"
                f"models/yolo/yolov8n-face-lindevs.pt"
            )

        print("Loading YOLO face detector...")

        self.model = YOLO(model_path)

        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

        print("YOLO face detector loaded successfully.")

    def detect(self, frame):
        """
        Detect faces in one OpenCV BGR frame.

        Returns:
            List of Detection objects.
        """

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        detections = []

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:

            confidence = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            detections.append(
                Detection(
                    bbox=(
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    confidence=confidence
                )
            )

        # Highest confidence first
        detections.sort(
            key=lambda detection: detection.confidence,
            reverse=True
        )

        return detections


# --------------------------------------------------
# Self-test
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting YOLO face detection self-test...")

    detector = FaceDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open laptop webcam."
        )

    frame_count = 0

    print("Opening webcam...")
    print("Running live face detection.")
    print("Press 'q' to quit.")

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame_count += 1

        detections = detector.detect(frame)

        # Draw detected faces
        for detection in detections:

            x1, y1, x2, y2 = detection.bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"Face "
                f"{detection.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Face count
        cv2.putText(
            frame,
            f"Faces: {len(detections)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "YOLO Face Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("\n================================")
    print("YOLO detection self-test complete")
    print("================================")
    print(f"Frames processed: {frame_count}")