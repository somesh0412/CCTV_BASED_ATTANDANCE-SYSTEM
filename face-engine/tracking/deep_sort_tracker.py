"""
tracking/deep_sort_tracker.py
-----------------------------------------------------------------------
Multi-face tracking using deep-sort-realtime.

Responsibilities (and ONLY these):
  - Take one frame's worth of YOLO face detections
  - Return stable Track IDs that persist across frames for the same
    physical face

The appearance embedder used here (MobileNet, bundled with
deep-sort-realtime) is ONLY for frame-to-frame tracking association.
It has nothing to do with student identity — that's FaceNet's job,
downstream in recognition/. Do not confuse the two.
-----------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import List, Tuple
from dataclasses import dataclass


import sys
from pathlib import Path

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.yolo_detector import Detection

@dataclass
class TrackedFace:
    """A single tracked face in the current frame."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords
    confidence: float

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class FaceTracker:
    """
    Wraps deep_sort_realtime.DeepSort for face tracking.

    Usage:
        tracker = FaceTracker()
        tracked_faces = tracker.update(frame, detections)
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        embedder_gpu: bool = False,
    ):
        """
        Args:
            max_age: frames a track survives with no matching detection
                     before being deleted (e.g. face turned away/occluded).
            n_init: consecutive detections required before a track is
                    "confirmed" and returned by update() — filters out
                    one-frame false positives from YOLO.
            max_iou_distance: max allowed IoU distance for a detection
                    to match an existing track's motion prediction.
            embedder_gpu: False since this environment is CPU-only.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder="mobilenet",
            embedder_gpu=embedder_gpu,
            bgr=True,  # our frames come from OpenCV (BGR), not RGB
        )

    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[TrackedFace]:
        """
        Args:
            frame: the current BGR frame (needed so DeepSORT can compute
                   appearance features for each detection).
            detections: this frame's YOLO face detections.

        Returns:
            List of TrackedFace for tracks confirmed and currently live.
            NOTE: bbox here is the tracker's Kalman-filtered estimate,
            which may differ slightly from the raw YOLO box — this is
            expected and is generally *more* stable for cropping.
        """
        # deep_sort_realtime expects: ([x, y, w, h], confidence, class_name)
        ds_detections = [
            ([d.bbox[0], d.bbox[1], d.width, d.height], d.confidence, "face")
            for d in detections
        ]

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        tracked_faces: List[TrackedFace] = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = track.to_ltrb()
            tracked_faces.append(
                TrackedFace(
                    track_id=int(track.track_id),
                    bbox=(int(l), int(t), int(r), int(b)),
                    confidence=float(track.det_conf) if track.det_conf is not None else 0.0,
                )
            )

        return tracked_faces


# ---------------------------------------------------------------------
# Self-test — runs YOLO + DeepSORT live on the webcam and draws boxes
# with persistent Track IDs. Depends on modules 2 and 3.
# Move around, leave frame and re-enter, and have a second person join
# to verify: IDs stay stable per-face, and distinct faces get distinct IDs.
# Press 'q' to quit.
# ---------------------------------------------------------------------
def _run_self_test() -> None:
    import time
    import sys
    from pathlib import Path
    import cv2

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from camera.webcam import WebcamStream
    from detection.yolo_detector import FaceDetector

    print("Loading YOLO face detector...")
    try:
        detector = FaceDetector()
    except FileNotFoundError as e:
        print(f"FAILED: {e}")
        return

    print("Initializing DeepSORT tracker...")
    tracker = FaceTracker()

    # Stable color per track ID so it's visually obvious an ID persists.
    id_colors = {}

    def color_for(track_id: int):
        if track_id not in id_colors:
            rng = np.random.default_rng(track_id)
            id_colors[track_id] = tuple(int(c) for c in rng.integers(80, 255, size=3))
        return id_colors[track_id]

    print("Opening webcam...")
    with WebcamStream(camera_index=0) as cam:
        print("Running live tracking. Press 'q' to quit.\n")

        frame_count = 0
        start_time = time.time()

        for frame in cam.frames():
            frame_count += 1
            detections = detector.detect(frame)
            tracked_faces = tracker.update(frame, detections)

            display_frame = frame.copy()
            for tf in tracked_faces:
                x1, y1, x2, y2 = tf.bbox
                color = color_for(tf.track_id)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {tf.track_id}"
                cv2.putText(
                    display_frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}  Tracked: {len(tracked_faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            cv2.imshow("DeepSORT Tracking Self-Test - press 'q' to quit", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"\nSelf-test complete. Processed {frame_count} frames.")
    print(f"Total unique track IDs assigned: {len(id_colors)}")


if __name__ == "__main__":
    _run_self_test()