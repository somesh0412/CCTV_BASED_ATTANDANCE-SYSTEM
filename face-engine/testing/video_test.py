import sys
import time
from pathlib import Path

import cv2

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Import EXISTING project modules
# --------------------------------------------------

from detection.yolo_detector import FaceDetector
from tracking.deep_sort_tracker import FaceTracker


# --------------------------------------------------
# Paths
# --------------------------------------------------

VIDEO_DIR = Path(__file__).resolve().parent / "videos"
RESULT_DIR = Path(__file__).resolve().parent / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Test one video
# --------------------------------------------------

def test_video(video_path: Path):

    print("\n" + "=" * 60)
    print(f"Testing: {video_path.name}")
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return

    # Video information
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = total_frames / input_fps if input_fps > 0 else 0

    print(f"Resolution       : {width} x {height}")
    print(f"Input FPS        : {input_fps:.2f}")
    print(f"Total frames     : {total_frames}")
    print(f"Duration         : {duration:.2f} seconds")

    # --------------------------------------------------
    # Create fresh YOLO + DeepSORT instances
    # --------------------------------------------------

    detector = FaceDetector(
        conf_threshold=0.5,
        imgsz=640,
        device="cpu"
    )

    tracker = FaceTracker(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        embedder_gpu=False
    )

    # --------------------------------------------------
    # Output video
    # --------------------------------------------------

    output_path = RESULT_DIR / f"{video_path.stem}_result.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        input_fps if input_fps > 0 else 20,
        (width, height)
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    frame_count = 0

    frames_with_faces = 0
    total_detections = 0

    total_tracked_faces = 0
    frames_with_tracks = 0

    unique_track_ids = set()

    processing_start = time.time()

    # --------------------------------------------------
    # Process video
    # --------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # -----------------------------
        # YOLO face detection
        # -----------------------------

        detections = detector.detect(frame)

        if len(detections) > 0:
            frames_with_faces += 1

        total_detections += len(detections)

        # -----------------------------
        # DeepSORT tracking
        # -----------------------------

        tracked_faces = tracker.update(
            frame,
            detections
        )

        if len(tracked_faces) > 0:
            frames_with_tracks += 1

        total_tracked_faces += len(tracked_faces)

        for track in tracked_faces:
            unique_track_ids.add(track.track_id)

        # -----------------------------
        # Draw YOLO detections
        # -----------------------------

        display_frame = frame.copy()

        for detection in detections:

            x1, y1, x2, y2 = detection.bbox

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"Face {detection.confidence:.2f}"

            cv2.putText(
                display_frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # -----------------------------
        # Draw DeepSORT tracks
        # -----------------------------

        for track in tracked_faces:

            x1, y1, x2, y2 = track.bbox

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            label = f"Track ID: {track.track_id}"

            cv2.putText(
                display_frame,
                label,
                (x1, min(y2 + 18, height - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        # -----------------------------
        # Display statistics on frame
        # -----------------------------

        cv2.putText(
            display_frame,
            f"Frame: {frame_count}/{total_frames}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Faces: {len(detections)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Tracks: {len(tracked_faces)}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # -----------------------------
        # Save output
        # -----------------------------

        writer.write(display_frame)

        # Optional preview
        cv2.imshow(
            "YOLO + DeepSORT Video Test",
            display_frame
        )

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nTest stopped by user.")
            break

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    processing_time = time.time() - processing_start

    # --------------------------------------------------
    # Final statistics
    # --------------------------------------------------

    processing_fps = (
        frame_count / processing_time
        if processing_time > 0
        else 0
    )

    average_faces = (
        total_detections / frame_count
        if frame_count > 0
        else 0
    )

    average_tracks = (
        total_tracked_faces / frame_count
        if frame_count > 0
        else 0
    )

    print("\n" + "-" * 60)
    print("TEST RESULTS")
    print("-" * 60)

    print(f"Video                 : {video_path.name}")
    print(f"Frames processed      : {frame_count}")
    print(f"Frames with faces     : {frames_with_faces}")
    print(f"Total face detections : {total_detections}")
    print(f"Average faces/frame   : {average_faces:.2f}")

    print(f"Frames with tracks    : {frames_with_tracks}")
    print(f"Total tracked faces   : {total_tracked_faces}")
    print(f"Unique Track IDs      : {len(unique_track_ids)}")
    print(f"Average tracks/frame  : {average_tracks:.2f}")

    print(f"Processing FPS        : {processing_fps:.2f}")

    print(f"\nResult saved to:")
    print(output_path)

    print("-" * 60)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n==============================================")
    print("YOLO + DeepSORT CLASSROOM VIDEO TEST")
    print("==============================================")

    video_files = sorted(
        VIDEO_DIR.glob("*.mp4")
    )

    if not video_files:
        print("\nNo MP4 videos found!")
        print(f"Put videos inside: {VIDEO_DIR}")
        return

    print(f"\nFound {len(video_files)} video(s).")

    for video_path in video_files:
        test_video(video_path)

    print("\n==============================================")
    print("ALL VIDEO TESTS COMPLETED")
    print("==============================================")


if __name__ == "__main__":
    main()