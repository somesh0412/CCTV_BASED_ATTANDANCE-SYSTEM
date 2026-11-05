from pathlib import Path
import sys
import json

import cv2


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.yolo_detector import FaceDetector


# --------------------------------------------------
# Paths
# --------------------------------------------------

TESTING_DIR = Path(__file__).resolve().parent

VIDEO1_DIR = TESTING_DIR / "ground_truth" / "video1"

ANNOTATION_FILE = VIDEO1_DIR / "annotations.json"

OUTPUT_DIR = VIDEO1_DIR / "comparison"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# IoU
# --------------------------------------------------

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n==============================================")
    print("YOLO VISUAL COMPARISON")
    print("==============================================")

    if not ANNOTATION_FILE.exists():

        print("annotations.json not found.")

        return

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        annotations = json.load(file)

    print(
        f"Frames to compare: {len(annotations)}"
    )

    # --------------------------------------------------
    # Load YOLO
    # --------------------------------------------------

    detector = FaceDetector(
        conf_threshold=0.5,
        imgsz=640,
        device="cpu"
    )

    # --------------------------------------------------
    # Process frames
    # --------------------------------------------------

    for frame_name, gt_boxes in annotations.items():

        image_path = VIDEO1_DIR / frame_name

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            continue

        detections = detector.detect(frame)

        detected_boxes = [
            list(detection.bbox)
            for detection in detections
        ]

        display = frame.copy()

        # --------------------------------------------------
        # Draw ground truth boxes
        # GREEN = manually annotated faces
        # --------------------------------------------------

        for i, gt_box in enumerate(gt_boxes):

            x1, y1, x2, y2 = gt_box

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"GT {i + 1}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

        # --------------------------------------------------
        # Draw YOLO boxes
        # RED = YOLO detections
        # --------------------------------------------------

        for i, det_box in enumerate(detected_boxes):

            x1, y1, x2, y2 = det_box

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                display,
                f"YOLO {i + 1}",
                (x1, min(y2 + 15, display.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )

        # --------------------------------------------------
        # Information
        # --------------------------------------------------

        cv2.putText(
            display,
            f"Ground Truth: {len(gt_boxes)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"YOLO: {len(detected_boxes)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        output_path = OUTPUT_DIR / frame_name

        cv2.imwrite(
            str(output_path),
            display
        )

        print(
            f"Saved comparison: {output_path.name}"
        )

    print("\n==============================================")
    print("COMPARISON COMPLETED")
    print("==============================================")

    print(f"\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()