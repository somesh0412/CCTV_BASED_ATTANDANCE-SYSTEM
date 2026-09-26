"""
face_processing/face_crop.py

Module 5: Face Crop Extraction

Flow:
    Webcam
       ↓
    YOLO Face Detection
       ↓
    DeepSORT Tracking
       ↓
    Tracked Face Bounding Box
       ↓
    Face Crop

This module ONLY extracts face crops.

It does NOT:
- run MediaPipe
- generate FaceNet embeddings
- perform recognition
- access MongoDB
"""

from typing import Optional, Tuple
from pathlib import Path
import sys

import cv2
import numpy as np


# ---------------------------------------------------------
# Make face-engine the project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Bounding box utilities
# ---------------------------------------------------------

def clip_bbox(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    """
    Keep bounding box inside the frame.

    bbox:
        (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), frame_width - 1))
    y1 = max(0, min(int(y1), frame_height - 1))

    x2 = max(0, min(int(x2), frame_width))
    y2 = max(0, min(int(y2), frame_height))

    return x1, y1, x2, y2


def pad_bbox(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding_ratio: float = 0.20,
) -> Tuple[int, int, int, int]:
    """
    Add padding around the detected face.

    0.20 means 20% of width/height is added
    to each side.
    """

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return 0, 0, 0, 0

    padding_x = int(width * padding_ratio)
    padding_y = int(height * padding_ratio)

    padded_bbox = (
        x1 - padding_x,
        y1 - padding_y,
        x2 + padding_x,
        y2 + padding_y,
    )

    return clip_bbox(
        padded_bbox,
        frame_width,
        frame_height,
    )


def make_square(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    """
    Make the bounding box approximately square.
    """

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return 0, 0, 0, 0

    if width == height:
        return clip_bbox(
            bbox,
            frame_width,
            frame_height,
        )

    if width > height:

        difference = width - height

        top = difference // 2
        bottom = difference - top

        y1 -= top
        y2 += bottom

    else:

        difference = height - width

        left = difference // 2
        right = difference - left

        x1 -= left
        x2 += right

    return clip_bbox(
        (x1, y1, x2, y2),
        frame_width,
        frame_height,
    )


# ---------------------------------------------------------
# Main face crop function
# ---------------------------------------------------------

def extract_face_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding_ratio: float = 0.20,
    square: bool = True,
    min_crop_size: int = 20,
) -> Optional[np.ndarray]:
    """
    Extract a face crop from an OpenCV BGR frame.

    Returns:
        BGR NumPy array if successful.
        None if the bounding box is invalid or too small.
    """

    if frame is None or frame.size == 0:
        return None

    frame_height, frame_width = frame.shape[:2]

    # 1. Clip original bbox
    bbox = clip_bbox(
        bbox,
        frame_width,
        frame_height,
    )

    # 2. Add padding
    bbox = pad_bbox(
        bbox,
        frame_width,
        frame_height,
        padding_ratio,
    )

    # 3. Make square
    if square:
        bbox = make_square(
            bbox,
            frame_width,
            frame_height,
        )

    x1, y1, x2, y2 = bbox

    # 4. Validate crop size
    crop_width = x2 - x1
    crop_height = y2 - y1

    if crop_width < min_crop_size or crop_height < min_crop_size:
        return None

    # 5. Extract crop
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Return a separate NumPy array
    return crop.copy()


# ---------------------------------------------------------
# Self-test
# ---------------------------------------------------------

def run_self_test():

    from camera.webcam import WebcamStream
    from detection.yolo_detector import FaceDetector
    from tracking.deep_sort_tracker import FaceTracker

    print("========================================")
    print("Module 5 - Face Crop Extraction")
    print("========================================")

    print("\nLoading YOLO face detector...")

    detector = FaceDetector()

    print("YOLO face detector loaded successfully.")

    print("\nInitializing DeepSORT tracker...")

    tracker = FaceTracker()

    print("DeepSORT tracker initialized.")

    # Folder for optional saved crops
    debug_dir = PROJECT_ROOT / "debug_crops"
    debug_dir.mkdir(
        exist_ok=True
    )

    print("\nOpening webcam...")

    frame_count = 0
    saved_count = 0

    with WebcamStream(camera_index=0) as camera:

        print(
            "Running face crop extraction."
        )

        print(
            "Press 's' to save current crops."
        )

        print(
            "Press 'q' to quit.\n"
        )

        for frame in camera.frames():

            frame_count += 1

            # ---------------------------------------------
            # YOLO
            # ---------------------------------------------

            detections = detector.detect(frame)

            # ---------------------------------------------
            # DeepSORT
            # ---------------------------------------------

            tracked_faces = tracker.update(
                frame,
                detections,
            )

            display_frame = frame.copy()

            crop_images = []

            # ---------------------------------------------
            # Extract crops
            # ---------------------------------------------

            for tracked_face in tracked_faces:

                x1, y1, x2, y2 = tracked_face.bbox

                # Draw tracking box
                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    display_frame,
                    f"ID {tracked_face.track_id}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Extract face crop
                crop = extract_face_crop(
                    frame,
                    tracked_face.bbox,
                )

                if crop is not None:

                    crop_images.append(
                        (
                            tracked_face.track_id,
                            crop,
                        )
                    )

            # ---------------------------------------------
            # Show crop thumbnails
            # ---------------------------------------------

            if crop_images:

                thumbnails = []

                thumbnail_height = 150

                for track_id, crop in crop_images:

                    crop_height, crop_width = crop.shape[:2]

                    scale = (
                        thumbnail_height
                        / crop_height
                    )

                    thumbnail_width = max(
                        1,
                        int(crop_width * scale),
                    )

                    thumbnail = cv2.resize(
                        crop,
                        (
                            thumbnail_width,
                            thumbnail_height,
                        ),
                    )

                    cv2.putText(
                        thumbnail,
                        f"ID {track_id}",
                        (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    thumbnails.append(
                        thumbnail
                    )

                # Make all thumbnails same width
                max_width = max(
                    image.shape[1]
                    for image in thumbnails
                )

                normalized_thumbnails = []

                for image in thumbnails:

                    height, width = image.shape[:2]

                    if width < max_width:

                        padding = np.zeros(
                            (
                                height,
                                max_width - width,
                                3,
                            ),
                            dtype=np.uint8,
                        )

                        image = np.hstack(
                            [
                                image,
                                padding,
                            ]
                        )

                    normalized_thumbnails.append(
                        image
                    )

                crop_display = np.hstack(
                    normalized_thumbnails
                )

                cv2.imshow(
                    "Extracted Face Crops",
                    crop_display,
                )

            # ---------------------------------------------
            # Information on main frame
            # ---------------------------------------------

            cv2.putText(
                display_frame,
                f"Faces: {len(crop_images)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                "S = Save crops | Q = Quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Face Crop Extraction",
                display_frame,
            )

            # ---------------------------------------------
            # Keyboard
            # ---------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):

                for track_id, crop in crop_images:

                    filename = (
                        f"track_{track_id}_"
                        f"frame_{frame_count}.jpg"
                    )

                    output_path = (
                        debug_dir / filename
                    )

                    cv2.imwrite(
                        str(output_path),
                        crop,
                    )

                    saved_count += 1

                print(
                    f"Saved {len(crop_images)} "
                    f"crop(s) to {debug_dir}"
                )

    cv2.destroyAllWindows()

    print("\n========================================")
    print("Face crop self-test complete")
    print("========================================")
    print(
        f"Frames processed: {frame_count}"
    )
    print(
        f"Crops saved:      {saved_count}"
    )
    print(
        f"Debug folder:     {debug_dir}"
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run_self_test()