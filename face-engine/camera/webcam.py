"""
camera/webcam.py
-----------------------------------------------------------------------
Thin wrapper around a local webcam (cv2.VideoCapture).

Responsibilities (and ONLY these):
  - Open/close the webcam device safely
  - Yield raw BGR frames (numpy arrays) to whoever asks

No detection, no tracking, no recognition — this file only knows how
to get frames out of the camera. camera/stream.py (CCTV/IP camera,
future work) is expected to expose the same frames()/read() interface
so the rest of the pipeline is source-agnostic.
-----------------------------------------------------------------------
"""

import time
from typing import Iterator, Optional

import cv2
import numpy as np


class WebcamStream:
    """
    Wraps cv2.VideoCapture for a local webcam.

    Usage:
        with WebcamStream() as cam:
            for frame in cam.frames():
                ...  # frame is a BGR numpy array (H, W, 3), uint8
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: Optional[int] = 30,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None

    # -------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------
    def open(self) -> "WebcamStream":
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.camera_index}. "
                "Is it connected, in use by another app, or is the "
                "index wrong (try 1 instead of 0, or vice versa)?"
            )

        # Requested — the camera may not honor these exactly; we read
        # back the actual values below so callers know the real state.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.target_fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        return self

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "WebcamStream":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # -------------------------------------------------------------
    # Frame access
    # -------------------------------------------------------------
    def read(self) -> Optional[np.ndarray]:
        """
        Grabs a single frame. Returns None if the read failed (e.g.
        camera disconnected) instead of raising, so a caller in a
        tight loop can decide whether to retry, skip, or stop.
        """
        if self.cap is None:
            raise RuntimeError("Camera is not open. Call open() or use 'with WebcamStream() as cam:'.")

        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """
        Generator that yields frames until the camera stops producing
        them. Skips (rather than stops on) occasional failed reads,
        since a single dropped frame from a webcam is normal.
        """
        consecutive_failures = 0
        max_consecutive_failures = 30  # ~1 second at 30fps before giving up

        while True:
            frame = self.read()

            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        f"Webcam stopped returning frames after "
                        f"{max_consecutive_failures} consecutive failed reads."
                    )
                continue

            consecutive_failures = 0
            yield frame

    # -------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------
    def get_actual_resolution(self) -> tuple:
        if self.cap is None:
            raise RuntimeError("Camera is not open.")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def get_actual_fps(self) -> float:
        if self.cap is None:
            raise RuntimeError("Camera is not open.")
        return self.cap.get(cv2.CAP_PROP_FPS)


# ---------------------------------------------------------------------
# Self-test — opens the webcam, prints actual resolution/FPS, and
# displays the raw feed. Press 'q' to quit.
# ---------------------------------------------------------------------
def _run_self_test() -> None:
    print("Opening webcam (index 0)...")

    try:
        with WebcamStream(camera_index=0) as cam:
            actual_w, actual_h = cam.get_actual_resolution()
            actual_fps = cam.get_actual_fps()

            print(f"Requested resolution: {cam.width}x{cam.height}")
            print(f"Actual resolution:    {actual_w}x{actual_h}")
            print(f"Actual FPS:            {actual_fps:.1f}")
            print("\nShowing raw webcam feed. Press 'q' in the window to quit.\n")

            frame_count = 0
            start_time = time.time()

            for frame in cam.frames():
                frame_count += 1
                elapsed = time.time() - start_time
                measured_fps = frame_count / elapsed if elapsed > 0 else 0.0

                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    f"Measured FPS: {measured_fps:.1f}  Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Webcam Self-Test - press 'q' to quit", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except RuntimeError as e:
        print(f"FAILED: {e}")
        return
    finally:
        cv2.destroyAllWindows()

    print(f"\nSelf-test complete. Captured {frame_count} frames.")


if __name__ == "__main__":
    _run_self_test()