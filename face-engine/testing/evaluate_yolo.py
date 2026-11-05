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


# --------------------------------------------------
# IoU calculation
# --------------------------------------------------

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)

    intersection_area = (
        intersection_width * intersection_height
    )

    # Areas
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# --------------------------------------------------
# Match YOLO detections with ground truth
# --------------------------------------------------

def match_boxes(
    ground_truth_boxes,
    detected_boxes,
    iou_threshold=0.5
):

    matches = []

    for gt_index, gt_box in enumerate(ground_truth_boxes):

        for det_index, det_box in enumerate(detected_boxes):

            iou = calculate_iou(
                gt_box,
                det_box
            )

            if iou >= iou_threshold:

                matches.append(
                    (
                        iou,
                        gt_index,
                        det_index
                    )
                )

    # Highest IoU first
    matches.sort(
        key=lambda x: x[0],
        reverse=True
    )

    matched_gt = set()
    matched_detection = set()

    true_positives = 0

    for iou, gt_index, det_index in matches:

        if gt_index in matched_gt:
            continue

        if det_index in matched_detection:
            continue

        matched_gt.add(gt_index)
        matched_detection.add(det_index)

        true_positives += 1

    false_negatives = (
        len(ground_truth_boxes)
        - true_positives
    )

    false_positives = (
        len(detected_boxes)
        - true_positives
    )

    return (
        true_positives,
        false_positives,
        false_negatives
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n==============================================")
    print("YOLO GROUND-TRUTH EVALUATION")
    print("==============================================")

    if not ANNOTATION_FILE.exists():

        print("\nERROR:")
        print(
            f"Annotation file not found:\n"
            f"{ANNOTATION_FILE}"
        )

        return

    # Load ground truth
    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        annotations = json.load(file)

    print(
        f"\nGround-truth frames: "
        f"{len(annotations)}"
    )

    # --------------------------------------------------
    # Load existing YOLO detector
    # --------------------------------------------------

    print("\nLoading YOLO detector...")

    detector = FaceDetector(
        conf_threshold=0.5,
        imgsz=640,
        device="cpu"
    )

    print("YOLO detector ready.")

    # --------------------------------------------------
    # Totals
    # --------------------------------------------------

    total_tp = 0
    total_fp = 0
    total_fn = 0

    iou_threshold = 0.5

    # --------------------------------------------------
    # Evaluate each frame
    # --------------------------------------------------

    for frame_name, gt_boxes in annotations.items():

        image_path = VIDEO1_DIR / frame_name

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:

            print(
                f"\nCould not read {frame_name}"
            )

            continue

        # YOLO
        detections = detector.detect(frame)

        detected_boxes = [
            list(detection.bbox)
            for detection in detections
        ]

        # Match
        tp, fp, fn = match_boxes(
            gt_boxes,
            detected_boxes,
            iou_threshold
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(
            f"{frame_name}: "
            f"GT={len(gt_boxes)} "
            f"YOLO={len(detected_boxes)} "
            f"TP={tp} "
            f"FP={fp} "
            f"FN={fn}"
        )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    precision_denominator = (
        total_tp + total_fp
    )

    recall_denominator = (
        total_tp + total_fn
    )

    if precision_denominator > 0:

        precision = (
            total_tp /
            precision_denominator
        )

    else:
        precision = 0.0

    if recall_denominator > 0:

        recall = (
            total_tp /
            recall_denominator
        )

    else:
        recall = 0.0

    if precision + recall > 0:

        f1 = (
            2 *
            precision *
            recall /
            (precision + recall)
        )

    else:
        f1 = 0.0

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n==============================================")
    print("FINAL YOLO EVALUATION")
    print("==============================================")

    print(f"Ground-truth faces : {total_tp + total_fn}")
    print(f"YOLO detections    : {total_tp + total_fp}")

    print(f"True Positives     : {total_tp}")
    print(f"False Positives    : {total_fp}")
    print(f"False Negatives    : {total_fn}")

    print(
        f"\nPrecision           : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall              : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1-score            : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"\nIoU threshold       : "
        f"{iou_threshold}"
    )

    print("==============================================")


if __name__ == "__main__":
    main()