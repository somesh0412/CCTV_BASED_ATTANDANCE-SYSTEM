"""Deterministic enrollment completion-policy tests; no webcam or model files."""

import unittest

import numpy as np

import enrollment.face_enrollment as enrollment_module
from enrollment.face_enrollment import (
    EnrollmentConfig,
    EnrollmentStatus,
    FaceEnrollmentManager,
)


class _Detection:
    bbox = (0, 0, 10, 10)
    confidence = 0.99


class _Detector:
    def detect(self, frame):
        return [_Detection()]


class _Embedder:
    def __init__(self):
        self.index = 0

    def get_embedding(self, crop):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[self.index % 512] = 1.0
        self.index += 1
        return embedding


class EnrollmentCompletionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.original_cropper = enrollment_module.extract_face_crop
        enrollment_module.extract_face_crop = lambda frame, bbox: frame
        self.frame = np.full((20, 20, 3), 100, dtype=np.uint8)

    def tearDown(self):
        enrollment_module.extract_face_crop = self.original_cropper

    def _manager(self):
        config = EnrollmentConfig(
            minimum_face_width=2,
            minimum_face_height=2,
            minimum_blur_score=0,
            minimum_sample_interval_seconds=0,
            max_duration_seconds=60,
        )
        return FaceEnrollmentManager(
            config,
            detector=_Detector(),
            embedder=_Embedder(),
        )

    def _collect(self, manager, count):
        manager.start_session()
        for _ in range(count):
            result = manager.process_frame(self.frame)
            self.assertTrue(result["accepted"])

    def _expire(self, manager):
        manager._started_at = -manager.config.max_duration_seconds - 1
        result = manager.process_frame(self.frame)
        self.assertFalse(result["accepted"])
        return result

    def test_thirty_samples_complete_immediately(self):
        manager = self._manager()
        self._collect(manager, 30)

        self.assertEqual(manager.get_progress()["current_status"], "completed")
        self.assertEqual(manager.get_progress()["completion_reason"], "target_reached")
        self.assertEqual(len(manager.get_embeddings()), 30)

    def test_twenty_eight_samples_complete_on_timeout_fallback(self):
        manager = self._manager()
        self._collect(manager, 28)
        result = self._expire(manager)

        self.assertEqual(result["progress"]["current_status"], "completed")
        self.assertEqual(result["progress"]["completion_reason"], "timeout_fallback")
        self.assertIn("timeout fallback", result["reason"])
        self.assertEqual(len(manager.get_embeddings()), 28)

    def test_twenty_five_samples_complete_on_timeout_fallback(self):
        manager = self._manager()
        self._collect(manager, 25)
        self._expire(manager)

        self.assertEqual(manager.get_progress()["current_status"], "completed")
        self.assertEqual(len(manager.get_embeddings()), 25)

    def test_twenty_four_samples_fail_on_timeout(self):
        manager = self._manager()
        self._collect(manager, 24)
        result = self._expire(manager)

        self.assertEqual(result["progress"]["current_status"], "failed")
        self.assertEqual(
            result["progress"]["completion_reason"],
            "timeout_below_minimum",
        )
        with self.assertRaises(RuntimeError):
            manager.get_embeddings()

    def test_cancelled_session_has_no_embeddings(self):
        manager = self._manager()
        self._collect(manager, 10)
        manager.cancel()

        self.assertEqual(manager.get_progress()["current_status"], "cancelled")
        with self.assertRaises(RuntimeError):
            manager.get_embeddings()


if __name__ == "__main__":
    unittest.main()
