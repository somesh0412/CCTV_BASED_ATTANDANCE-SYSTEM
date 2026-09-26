"""
pipeline/face_pipeline.py

Module 11: Complete Face Pipeline

Flow:

    Webcam
       ↓
    YOLO
       ↓
    DeepSORT
       ↓
    Face Crop
       ↓
    Identity Cache
       │
       ├── Known Track → use cached identity
       │
       └── New Track
              ↓
           FaceNet
              ↓
           Matcher
              ↓
        Student Identity
              ↓
        Identity Cache

No attendance recording yet.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# EXISTING MODULES
# =========================================================

from camera.webcam import WebcamStream
from detection.yolo_detector import FaceDetector
from tracking.deep_sort_tracker import FaceTracker
from face_processing.face_crop import extract_face_crop

from recognition.embedding import FaceEmbedder
from recognition.matcher import FaceMatcher

from database.student_repository import StudentRepository
from identity.identity_cache import IdentityCache


# =========================================================
# PIPELINE RESULT
# =========================================================

@dataclass
class PipelineResult:
    """
    Result produced for one tracked face.
    """

    track_id: int

    # DeepSORT bounding box
    bbox: tuple

    # Student information
    student_id: str
    name: str

    # Similarity returned by Matcher.
    # For cached identities this is 0.0 because
    # FaceNet/Matcher was not executed again.
    similarity: float

    # True when identity came from Identity Cache.
    from_cache: bool


# =========================================================
# FACE PIPELINE
# =========================================================

class FacePipeline:

    def __init__(
        self,
        matcher_threshold: float = 0.40,
        cache_expiry_seconds: float = 3.0,
    ):

        print("========================================")
        print("Initializing Face Pipeline")
        print("========================================")

        # -------------------------------------------------
        # YOLO
        # -------------------------------------------------

        print("\nLoading YOLO face detector...")

        self.detector = FaceDetector()

        print(
            "YOLO face detector loaded successfully."
        )

        # -------------------------------------------------
        # DeepSORT
        # -------------------------------------------------

        print("\nInitializing DeepSORT...")

        self.tracker = FaceTracker()

        print(
            "DeepSORT initialized successfully."
        )

        # -------------------------------------------------
        # FaceNet
        # -------------------------------------------------

        print("\nLoading FaceNet...")

        self.embedder = FaceEmbedder()

        print(
            "FaceNet loaded successfully."
        )

        # -------------------------------------------------
        # Student Repository
        # -------------------------------------------------

        print("\nLoading student repository...")

        self.repository = StudentRepository()

        self.repository.load()

        print(
            f"Students loaded: "
            f"{self.repository.student_count()}"
        )

        print(
            f"Embeddings loaded: "
            f"{self.repository.embedding_count()}"
        )

        # -------------------------------------------------
        # Matcher
        # -------------------------------------------------

        print("\nInitializing matcher...")

        self.matcher = FaceMatcher(
            self.repository,
            threshold=matcher_threshold,
        )

        print(
            f"Matcher threshold: "
            f"{matcher_threshold:.2f}"
        )

        # -------------------------------------------------
        # Identity Cache
        # -------------------------------------------------

        print("\nInitializing identity cache...")

        self.identity_cache = IdentityCache(
            expiry_seconds=cache_expiry_seconds
        )

        print(
            f"Cache expiry: "
            f"{cache_expiry_seconds:.1f} seconds"
        )

        print("\n========================================")
        print("Face Pipeline initialized successfully")
        print("========================================")

    # =====================================================
    # PROCESS ONE FRAME
    # =====================================================

    def process_frame(self, frame):
        """
        Process one webcam frame.

        Important:

        YOLO and DeepSORT are called ONLY ONCE per frame.

        Returns:
            List[PipelineResult]
        """

        results = []

        current_time = time.time()

        # -------------------------------------------------
        # STEP 1: YOLO DETECTION
        # -------------------------------------------------

        detections = self.detector.detect(frame)

        # -------------------------------------------------
        # STEP 2: DEEPSORT TRACKING
        # -------------------------------------------------

        tracked_faces = self.tracker.update(
            frame,
            detections,
        )

        # -------------------------------------------------
        # STEP 3: PROCESS EACH TRACK
        # -------------------------------------------------

        for tracked_face in tracked_faces:

            track_id = tracked_face.track_id

            bbox = tracked_face.bbox

            # -------------------------------------------------
            # STEP 3A: CHECK IDENTITY CACHE
            # -------------------------------------------------

            cached_identity = (
                self.identity_cache.get_identity(
                    track_id,
                    timestamp=current_time,
                )
            )

            # -------------------------------------------------
            # EXISTING TRACK
            # -------------------------------------------------

            if cached_identity is not None:

                # Refresh last_seen
                self.identity_cache.mark_seen(
                    track_id,
                    timestamp=current_time,
                )

                results.append(
                    PipelineResult(
                        track_id=track_id,
                        bbox=bbox,
                        student_id=(
                            cached_identity.student_id
                        ),
                        name=cached_identity.name,
                        similarity=0.0,
                        from_cache=True,
                    )
                )

                # IMPORTANT:
                # No FaceNet.
                # No Matcher.
                # Continue to next track.
                continue

            # -------------------------------------------------
            # NEW TRACK
            # -------------------------------------------------

            crop = extract_face_crop(
                frame,
                bbox,
            )

            if crop is None:
                continue

            # -------------------------------------------------
            # STEP 4: FACENET
            # -------------------------------------------------

            embedding = (
                self.embedder.get_embedding(
                    crop
                )
            )

            if embedding is None:
                continue

            # -------------------------------------------------
            # STEP 5: MATCHER
            # -------------------------------------------------

            match = self.matcher.match(
                embedding
            )

            if match is None:
                continue

            # -------------------------------------------------
            # UNKNOWN FACE
            # -------------------------------------------------

            if not match.matched:

                results.append(
                    PipelineResult(
                        track_id=track_id,
                        bbox=bbox,
                        student_id="UNKNOWN",
                        name="Unknown",
                        similarity=match.similarity,
                        from_cache=False,
                    )
                )

                continue

            # -------------------------------------------------
            # KNOWN STUDENT
            # -------------------------------------------------

            self.identity_cache.set_identity(
                track_id=track_id,
                student_id=match.student_id,
                name=match.name,
                timestamp=current_time,
            )

            results.append(
                PipelineResult(
                    track_id=track_id,
                    bbox=bbox,
                    student_id=match.student_id,
                    name=match.name,
                    similarity=match.similarity,
                    from_cache=False,
                )
            )

        # -------------------------------------------------
        # STEP 6: CLEAN OLD CACHE ENTRIES
        # -------------------------------------------------

        self.identity_cache.cleanup(
            timestamp=current_time
        )

        return results


# =========================================================
# DRAW RESULTS
# =========================================================

def draw_pipeline_results(
    frame,
    results,
):
    """
    Draw bounding boxes and identity information.
    """

    display_frame = frame.copy()

    for result in results:

        x1, y1, x2, y2 = result.bbox

        # -------------------------------------------------
        # Choose display text
        # -------------------------------------------------

        if result.from_cache:

            label = (
                f"ID {result.track_id} | "
                f"{result.student_id} | "
                f"{result.name} | CACHED"
            )

        elif result.student_id == "UNKNOWN":

            label = (
                f"ID {result.track_id} | "
                f"UNKNOWN | "
                f"{result.similarity:.3f}"
            )

        else:

            label = (
                f"ID {result.track_id} | "
                f"{result.student_id} | "
                f"{result.name} | "
                f"{result.similarity:.3f}"
            )

        # -------------------------------------------------
        # Draw bounding box
        # -------------------------------------------------

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # -------------------------------------------------
        # Draw label
        # -------------------------------------------------

        text_y = max(
            25,
            y1 - 10,
        )

        cv2.putText(
            display_frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    return display_frame


# =========================================================
# LIVE SELF-TEST
# =========================================================

def run_live_test():

    print("\n========================================")
    print("Starting Live Face Pipeline")
    print("========================================")

    # -----------------------------------------------------
    # Initialize pipeline
    # -----------------------------------------------------

    pipeline = FacePipeline(
        matcher_threshold=0.40,
        cache_expiry_seconds=3.0,
    )

    print("\nOpening webcam...")

    frame_count = 0

    # These counters are only for observing behavior.
    new_identity_results = 0
    cached_results = 0

    # -----------------------------------------------------
    # Open webcam
    # -----------------------------------------------------

    with WebcamStream(
        camera_index=0
    ) as camera:

        print(
            "\n========================================"
        )

        print(
            "Running complete face pipeline."
        )

        print(
            "Press 'q' to quit."
        )

        print(
            "========================================\n"
        )

        # -------------------------------------------------
        # Process frames
        # -------------------------------------------------

        for frame in camera.frames():

            frame_count += 1

            # -------------------------------------------------
            # Run COMPLETE pipeline
            # -------------------------------------------------

            results = pipeline.process_frame(
                frame
            )

            # -------------------------------------------------
            # Count cache usage
            # -------------------------------------------------

            for result in results:

                if result.from_cache:

                    cached_results += 1

                else:

                    new_identity_results += 1

            # -------------------------------------------------
            # Draw results
            # -------------------------------------------------

            display_frame = (
                draw_pipeline_results(
                    frame,
                    results,
                )
            )

            # -------------------------------------------------
            # Information panel
            # -------------------------------------------------

            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Tracks: {len(results)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                "Q = Quit",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # -------------------------------------------------
            # Show window
            # -------------------------------------------------

            cv2.imshow(
                "Complete Face Pipeline",
                display_frame,
            )

            # -------------------------------------------------
            # Keyboard
            # -------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    cv2.destroyAllWindows()

    print("\n========================================")
    print("Face Pipeline Self-Test Complete")
    print("========================================")

    print(
        f"Frames processed: "
        f"{frame_count}"
    )

    print(
        f"New/recognition results: "
        f"{new_identity_results}"
    )

    print(
        f"Cached results: "
        f"{cached_results}"
    )

    print("\nPipeline test finished.")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_live_test()