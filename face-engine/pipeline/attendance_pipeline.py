"""
pipeline/attendance_pipeline.py

Module 13: Face Recognition + Attendance Integration

Connects:

    Module 11 - FacePipeline
                    ↓
    Module 12 - AttendanceEngine

Responsibilities:
    - Capture webcam frames
    - Run the existing FacePipeline
    - Send recognized students to AttendanceEngine
    - Prevent duplicate attendance
    - Display recognition and attendance status

This module does NOT:
    - Access MongoDB directly
    - Run YOLO directly
    - Run FaceNet directly
    - Implement matching logic
"""

import sys
from pathlib import Path

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
from pipeline.face_pipeline import FacePipeline
from attendance.attendance_engine import AttendanceEngine
from backend.attendance_api import AttendanceAPI


# =========================================================
# INTEGRATED PIPELINE
# =========================================================

class AttendancePipeline:

    def __init__(self):

        print("========================================")
        print("Initializing Attendance Pipeline")
        print("========================================")

        # -------------------------------------------------
        # Module 11
        # -------------------------------------------------

        print("\nLoading Face Pipeline...")

        self.face_pipeline = FacePipeline(
            matcher_threshold=0.40,
            cache_expiry_seconds=3.0,
        )

        print(
            "Face Pipeline loaded successfully."
        )

        # -------------------------------------------------
        # Module 12
        # -------------------------------------------------

        print("\nLoading Attendance Engine...")

        self.attendance_engine = AttendanceEngine()
        self.attendance_api = AttendanceAPI(
            base_url="http://localhost:5000",
            endpoint="/api/attendance",
        )

        print(
            "Attendance Engine loaded successfully."
        )

        print("\n========================================")
        print("Attendance Pipeline ready")
        print("========================================")

    # =====================================================
    # PROCESS ONE FRAME
    # =====================================================

    def process_frame(self, frame):

        """
        Process one webcam frame.

        Returns:
            face_results
            newly_created_attendance_events
        """

        # -------------------------------------------------
        # Run Module 11
        # -------------------------------------------------

        face_results = (
            self.face_pipeline.process_frame(
                frame
            )
        )

        new_attendance_events = []

        # -------------------------------------------------
        # Send recognized students to Module 12
        # -------------------------------------------------

        for result in face_results:

            # -------------------------------------------------
            # Ignore unknown faces
            # -------------------------------------------------

            if (
                result.student_id == "UNKNOWN"
            ):
                continue

            # -------------------------------------------------
            # Mark attendance
            # -------------------------------------------------

            event = self.attendance_engine.mark_attendance(
                student_id=result.student_id,
                name=result.name,
                track_id=result.track_id,
                similarity=result.similarity,
            )

            if event is not None:

                new_attendance_events.append(event)

                self.attendance_api.send_event(event)

        return (
            face_results,
            new_attendance_events,
        )


# =========================================================
# DRAW RESULTS
# =========================================================

def draw_results(
    frame,
    face_results,
    new_attendance_events,
    attendance_engine,
):

    display_frame = frame.copy()

    # IDs of students whose attendance was newly created
    newly_marked_ids = {
        event.student_id
        for event in new_attendance_events
    }

    # -----------------------------------------------------
    # Draw recognized faces
    # -----------------------------------------------------

    for result in face_results:

        x1, y1, x2, y2 = result.bbox

        # -------------------------------------------------
        # Determine display state
        # -------------------------------------------------

        if result.student_id == "UNKNOWN":

            label = (
                f"ID {result.track_id} | UNKNOWN"
            )

        elif result.student_id in newly_marked_ids:

            label = (
                f"ID {result.track_id} | "
                f"{result.student_id} | "
                f"{result.name} | "
                f"ATTENDANCE MARKED"
            )

        elif result.from_cache:

            label = (
                f"ID {result.track_id} | "
                f"{result.student_id} | "
                f"{result.name} | "
                f"PRESENT"
            )

        else:

            label = (
                f"ID {result.track_id} | "
                f"{result.student_id} | "
                f"{result.name}"
            )

        # -------------------------------------------------
        # Draw box
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

        cv2.putText(
            display_frame,
            label,
            (
                x1,
                max(25, y1 - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    # -----------------------------------------------------
    # Attendance counter
    # -----------------------------------------------------

    attendance_count = (
        attendance_engine.attendance_count()
    )

    cv2.putText(
        display_frame,
        f"Present: {attendance_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display_frame,
        "Press Q to quit",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    return display_frame


# =========================================================
# LIVE SELF-TEST
# =========================================================

def run_live_test():

    print("\n========================================")
    print("Module 13 - Attendance Integration Test")
    print("========================================")

    # -----------------------------------------------------
    # Initialize integrated pipeline
    # -----------------------------------------------------

    pipeline = AttendancePipeline()

    print("\nOpening webcam...")

    frame_count = 0
    total_new_attendance = 0

    # -----------------------------------------------------
    # Webcam
    # -----------------------------------------------------

    with WebcamStream(
        camera_index=0
    ) as camera:

        print(
            "\nRunning recognition + attendance."
        )

        print(
            "Look at the camera."
        )

        print(
            "Press 'q' to quit.\n"
        )

        # -------------------------------------------------
        # Frame loop
        # -------------------------------------------------

        for frame in camera.frames():

            frame_count += 1

            # -------------------------------------------------
            # Process frame
            # -------------------------------------------------

            (
                face_results,
                new_events,
            ) = pipeline.process_frame(
                frame
            )

            # -------------------------------------------------
            # New attendance events
            # -------------------------------------------------

            for event in new_events:

                total_new_attendance += 1

                print(
                    "\n========================================"
                )

                print(
                    "ATTENDANCE EVENT CREATED"
                )

                print(
                    "========================================"
                )

                print(
                    f"Student ID : {event.student_id}"
                )

                print(
                    f"Name       : {event.name}"
                )

                print(
                    f"Date       : {event.date}"
                )

                print(
                    f"Time       : {event.time}"
                )

                print(
                    f"Status     : {event.status}"
                )

                print(
                    f"Track ID   : {event.track_id}"
                )

                print(
                    f"Similarity : {event.similarity:.4f}"
                )

                print(
                    "========================================\n"
                )

            # -------------------------------------------------
            # Draw
            # -------------------------------------------------

            display_frame = draw_results(
                frame,
                face_results,
                new_events,
                pipeline.attendance_engine,
            )

            # -------------------------------------------------
            # Show
            # -------------------------------------------------

            cv2.imshow(
                "CCTV Attendance System",
                display_frame,
            )

            # -------------------------------------------------
            # Quit
            # -------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    cv2.destroyAllWindows()

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n========================================")
    print("Module 13 Test Complete")
    print("========================================")

    print(
        f"Frames processed: "
        f"{frame_count}"
    )

    print(
        f"New attendance events: "
        f"{total_new_attendance}"
    )

    print(
        f"Students marked present: "
        f"{pipeline.attendance_engine.attendance_count()}"
    )

    print("\nAttendance records:")

    for event in (
        pipeline.attendance_engine
        .get_all_attendance()
    ):

        print(
            f"  {event.student_id:<10}"
            f"{event.name:<15}"
            f"{event.date} "
            f"{event.time} "
            f"{event.status}"
        )

    print("\n========================================")
    print("Integration test finished")
    print("========================================")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_live_test()