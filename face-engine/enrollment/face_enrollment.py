"""Capture quality-checked face embeddings for one enrollment session.

This module deliberately does not persist data or access MongoDB.  It reuses
the face detector, cropper, and FaceNet embedder already used by the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

import cv2
import numpy as np

from face_processing.face_crop import extract_face_crop

if TYPE_CHECKING:
    from detection.yolo_detector import FaceDetector
    from recognition.embedding import FaceEmbedder


LOGGER = logging.getLogger(__name__)
EMBEDDING_DIMENSION = 512


class EnrollmentStatus(str, Enum):
    """Lifecycle states for an enrollment session."""

    IDLE = "idle"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class EnrollmentConfig:
    """Configurable quality and session limits.

    The defaults are practical starting points, not scientifically optimal
    thresholds.  They should be tuned for the enrollment camera and lighting.
    """

    target_embeddings: int = 30
    minimum_accepted_embeddings: int = 25
    max_duration_seconds: float = 180.0
    minimum_detection_confidence: float = 0.5
    minimum_face_width: int = 80
    minimum_face_height: int = 80
    minimum_brightness: float = 45.0
    maximum_brightness: float = 220.0
    minimum_blur_score: float = 80.0
    minimum_sample_interval_seconds: float = 0.5
    duplicate_similarity_threshold: float = 0.985

    def __post_init__(self) -> None:
        if self.target_embeddings <= 0:
            raise ValueError("target_embeddings must be positive")
        if not 0 < self.minimum_accepted_embeddings <= self.target_embeddings:
            raise ValueError(
                "minimum_accepted_embeddings must be between 1 and target_embeddings"
            )
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        if not 0 <= self.minimum_detection_confidence <= 1:
            raise ValueError("minimum_detection_confidence must be between 0 and 1")
        if self.minimum_face_width <= 0 or self.minimum_face_height <= 0:
            raise ValueError("minimum face dimensions must be positive")
        if (
            self.minimum_brightness < 0
            or self.maximum_brightness < 0
            or self.minimum_brightness > 255
            or self.maximum_brightness > 255
        ):
            raise ValueError("brightness thresholds must be within 0..255")
        if self.minimum_brightness >= self.maximum_brightness:
            raise ValueError("minimum_brightness must be below maximum_brightness")
        if self.minimum_blur_score < 0:
            raise ValueError("minimum_blur_score cannot be negative")
        if self.minimum_sample_interval_seconds < 0:
            raise ValueError("minimum_sample_interval_seconds cannot be negative")
        if not 0 <= self.duplicate_similarity_threshold <= 1:
            raise ValueError("duplicate_similarity_threshold must be between 0 and 1")


class FaceEnrollmentManager:
    """Manage one in-memory enrollment session at a time."""

    def __init__(
        self,
        config: Optional[EnrollmentConfig] = None,
        *,
        device: str = "cpu",
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
    ) -> None:
        self.config = config or EnrollmentConfig()

        try:
            if detector is None:
                from detection.yolo_detector import FaceDetector as Detector

                detector = Detector(
                    conf_threshold=self.config.minimum_detection_confidence,
                    device=device,
                )
            if embedder is None:
                from recognition.embedding import FaceEmbedder as Embedder

                embedder = Embedder(device=device)
            self.detector = (
                detector
            )
            self.embedder = (
                embedder
            )
        except (FileNotFoundError, ImportError, OSError, RuntimeError) as error:
            LOGGER.exception("Could not initialize enrollment models")
            raise RuntimeError(f"Enrollment model initialization failed: {error}") from error

        self._status = EnrollmentStatus.IDLE
        self._started_at: Optional[float] = None
        self._last_accepted_at: Optional[float] = None
        self._embeddings: list[np.ndarray] = []
        self._last_rejection_reason: Optional[str] = None
        self._completion_reason: Optional[str] = None

    def start_session(self) -> dict[str, Any]:
        """Start or reset the in-memory enrollment session."""
        self._status = EnrollmentStatus.ACTIVE
        self._started_at = time.monotonic()
        self._last_accepted_at = None
        self._embeddings = []
        self._last_rejection_reason = None
        self._completion_reason = None
        return self.get_progress()

    def reset_session(self) -> dict[str, Any]:
        """Reset the session and immediately start a fresh enrollment."""
        return self.start_session()

    def cancel(self) -> dict[str, Any]:
        """Cancel the active session and discard its collected embeddings."""
        if self._status == EnrollmentStatus.ACTIVE:
            self._status = EnrollmentStatus.CANCELLED
            self._embeddings = []
            self._completion_reason = None
        return self.get_progress()

    def has_reached_target(self) -> bool:
        """Return whether the target count has been accepted."""
        return self._status == EnrollmentStatus.COMPLETED

    def process_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Process one BGR frame and return an explicit acceptance result."""
        if self._status != EnrollmentStatus.ACTIVE:
            return self._rejected_result(
                "Enrollment is not active. Start a session before submitting frames."
            )

        if self._has_timed_out():
            return self._timeout_result()

        if not self._valid_frame(frame_bgr):
            return self._rejected_result(
                "Invalid frame: expected a non-empty BGR image with three channels."
            )

        now = time.monotonic()
        if (
            self._last_accepted_at is not None
            and now - self._last_accepted_at
            < self.config.minimum_sample_interval_seconds
        ):
            return self._rejected_result(
                "Please hold still briefly; waiting for the next sample interval."
            )

        try:
            detections = self.detector.detect(frame_bgr)
        except (RuntimeError, ValueError, cv2.error) as error:
            LOGGER.exception("Face detection failed for an enrollment frame")
            return self._rejected_result(f"Face detection failed: {error}")

        if not detections:
            return self._rejected_result(
                "No face detected. Move into view and keep your face centered."
            )
        if len(detections) > 1:
            return self._rejected_result(
                "Multiple faces detected. Ensure only the enrolling student is visible."
            )

        detection = detections[0]
        if detection.confidence < self.config.minimum_detection_confidence:
            return self._rejected_result(
                "Face detection confidence is too low. Move closer and face the camera."
            )

        crop = extract_face_crop(frame_bgr, detection.bbox)
        if crop is None or not self._valid_crop(crop):
            return self._rejected_result(
                "Face is too small or the crop is invalid. Move a little closer."
            )

        quality_reason = self._quality_rejection_reason(crop)
        if quality_reason is not None:
            return self._rejected_result(quality_reason)

        try:
            embedding = self.embedder.get_embedding(crop)
        except (RuntimeError, ValueError, cv2.error) as error:
            LOGGER.exception("Embedding generation failed for an enrollment frame")
            return self._rejected_result(f"Embedding generation failed: {error}")

        validated_embedding = self._validate_embedding(embedding)
        if validated_embedding is None:
            return self._rejected_result(
                "The face embedding is invalid. Please try another frame."
            )

        if self._is_duplicate(validated_embedding):
            return self._rejected_result(
                "This sample is too similar to a recent sample. Slowly vary your position."
            )

        self._embeddings.append(validated_embedding)
        self._last_accepted_at = now
        self._last_rejection_reason = None
        if len(self._embeddings) >= self.config.target_embeddings:
            self._status = EnrollmentStatus.COMPLETED
            self._completion_reason = "target_reached"

        return {
            "accepted": True,
            "reason": None,
            "progress": self.get_progress(),
        }

    def get_progress(self) -> dict[str, Any]:
        """Return JSON-friendly progress for a frontend or self-test."""
        count = len(self._embeddings)
        target = self.config.target_embeddings
        percentage = min(100.0, count * 100.0 / target)
        return {
            "accepted_sample_count": count,
            "target_sample_count": target,
            "percentage_completed": round(percentage, 2),
            "current_status": self._status.value,
            "last_rejection_reason": self._last_rejection_reason,
            "completion_reason": self._completion_reason,
            "suggested_student_instruction": self._instruction_for_count(count),
        }

    def get_embeddings(self) -> list[np.ndarray]:
        """Return copies of embeddings after successful completion only."""
        if self._status != EnrollmentStatus.COMPLETED:
            raise RuntimeError("Embeddings are available only after successful completion")
        return [embedding.copy() for embedding in self._embeddings]

    def _valid_frame(self, frame: Any) -> bool:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return False
        if frame.shape[2] != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            return False
        if not np.issubdtype(frame.dtype, np.number):
            return False
        try:
            return bool(np.isfinite(frame).all())
        except (TypeError, ValueError):
            return False

    def _valid_crop(self, crop: np.ndarray) -> bool:
        return (
            crop.ndim == 3
            and crop.shape[2] == 3
            and crop.shape[1] >= self.config.minimum_face_width
            and crop.shape[0] >= self.config.minimum_face_height
        )

    def _quality_rejection_reason(self, crop: np.ndarray) -> Optional[str]:
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(grayscale))
        if brightness < self.config.minimum_brightness:
            return "Lighting is too dark. Move to a brighter, evenly lit area."
        if brightness > self.config.maximum_brightness:
            return "Lighting is too bright. Avoid glare and direct light."

        blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        if blur_score < self.config.minimum_blur_score:
            return "The face crop is blurry. Hold the camera and your head still."
        return None

    def _validate_embedding(self, embedding: Any) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        try:
            values = np.asarray(embedding)
        except (TypeError, ValueError):
            return None
        if values.size != EMBEDDING_DIMENSION or not np.issubdtype(
            values.dtype, np.number
        ):
            return None
        values = values.reshape(EMBEDDING_DIMENSION)
        if not np.isfinite(values).all():
            return None
        return values.astype(np.float32, copy=True)

    def _is_duplicate(self, candidate: np.ndarray) -> bool:
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm == 0:
            return True
        for existing in self._embeddings:
            existing_norm = float(np.linalg.norm(existing))
            if existing_norm == 0:
                continue
            similarity = float(
                np.dot(candidate, existing) / (candidate_norm * existing_norm)
            )
            if similarity >= self.config.duplicate_similarity_threshold:
                return True
        return False

    def _has_timed_out(self) -> bool:
        if self._started_at is None:
            return False
        if time.monotonic() - self._started_at <= self.config.max_duration_seconds:
            return False
        if len(self._embeddings) < self.config.minimum_accepted_embeddings:
            self._status = EnrollmentStatus.FAILED
            self._completion_reason = "timeout_below_minimum"
        else:
            self._status = EnrollmentStatus.COMPLETED
            self._completion_reason = "timeout_fallback"
        return True

    def _timeout_result(self) -> dict[str, Any]:
        if self._status == EnrollmentStatus.FAILED:
            reason = (
                "Enrollment timed out before the minimum number of valid samples "
                f"was collected ({len(self._embeddings)}/"
                f"{self.config.minimum_accepted_embeddings}); enrollment failed."
            )
        else:
            reason = (
                f"Enrollment completed with {len(self._embeddings)} valid samples "
                "after timeout fallback. The target of "
                f"{self.config.target_embeddings} was not reached."
            )
        self._last_rejection_reason = reason
        return {
            "accepted": False,
            "reason": reason,
            "progress": self.get_progress(),
        }

    def _rejected_result(self, reason: str) -> dict[str, Any]:
        self._last_rejection_reason = reason
        return {
            "accepted": False,
            "reason": reason,
            "progress": self.get_progress(),
        }

    @staticmethod
    def _instruction_for_count(count: int) -> str:
        if count < 8:
            return "Look straight at the camera and keep your face centered."
        if count < 18:
            return "Slowly vary your head slightly left and right; no sudden movements."
        if count < 24:
            return "Slowly vary your head slightly up and down while staying visible."
        if count < 27:
            return "Make a few natural expressions while keeping your face in view."
        if count < 30:
            return "Change your distance slightly, then return to a clear view."
        return "Enrollment complete."


__all__ = [
    "EMBEDDING_DIMENSION",
    "EnrollmentConfig",
    "EnrollmentStatus",
    "FaceEnrollmentManager",
]
