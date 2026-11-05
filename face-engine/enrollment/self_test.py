"""Opt-in webcam self-test for the face enrollment manager.

Run from the face-engine directory with:
    python -m enrollment.self_test
"""

import cv2

from enrollment.face_enrollment import FaceEnrollmentManager, EnrollmentStatus


def main() -> None:
    manager = FaceEnrollmentManager()
    manager.start_session()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    print("Enrollment self-test started. Press 'q' or Escape to cancel.")
    try:
        while manager.get_progress()["current_status"] == EnrollmentStatus.ACTIVE.value:
            success, frame = camera.read()
            if not success:
                print("Could not read a webcam frame.")
                break

            result = manager.process_frame(frame)
            progress = result["progress"]
            status = progress["current_status"]
            reason = progress["last_rejection_reason"] or "Sample accepted."

            cv2.putText(
                frame,
                f"Samples: {progress['accepted_sample_count']}/{progress['target_sample_count']}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                progress["suggested_student_instruction"],
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                reason[:100],
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 180, 255),
                2,
            )
            cv2.imshow("Face Enrollment Self-Test", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                manager.cancel()
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    progress = manager.get_progress()
    print(
        f"Enrollment ended with status '{progress['current_status']}' "
        f"and {progress['accepted_sample_count']} samples."
    )
    if manager.has_reached_target():
        embeddings = manager.get_embeddings()
        print(f"Ready embeddings: {len(embeddings)} x {embeddings[0].shape[0]} float32")
    else:
        print("No embeddings were written to MongoDB or disk.")


if __name__ == "__main__":
    main()
