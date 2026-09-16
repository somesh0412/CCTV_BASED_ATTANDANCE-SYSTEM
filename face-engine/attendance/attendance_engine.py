"""
attendance/attendance_engine.py
-----------------------------------------------------------------------

Module 12: Attendance Engine

Responsibilities:
    - Receive recognized student identities from Face Pipeline
    - Prevent duplicate attendance for the same student
    - Create attendance events
    - Keep attendance decision logic separate from MongoDB and Node.js

This module DOES NOT:
    - Run YOLO
    - Run DeepSORT
    - Run FaceNet
    - Match faces
    - Directly write to MongoDB
    - Call the Node.js backend

Later architecture:

    Face Pipeline
          ↓
    Attendance Engine
          ↓
    Attendance Event
          ↓
    Node.js Backend
          ↓
    MongoDB
          ↓
    Teacher Dashboard

-----------------------------------------------------------------------
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional


# =========================================================
# ATTENDANCE EVENT
# =========================================================

@dataclass
class AttendanceEvent:
    """
    Represents one attendance event.

    This is the object that will eventually be sent to
    the Node.js backend.
    """

    student_id: str
    name: str

    date: str
    time: str

    status: str

    track_id: int

    similarity: float


# =========================================================
# ATTENDANCE ENGINE
# =========================================================

class AttendanceEngine:

    def __init__(self):
        """
        Creates a new attendance engine.

        Attendance is currently stored in memory.

        Example:

            marked_students = {
                "STU008": AttendanceEvent(...),
                "STU003": AttendanceEvent(...)
            }
        """

        self.marked_students: Dict[str, AttendanceEvent] = {}

    # =====================================================
    # MARK ATTENDANCE
    # =====================================================

    def mark_attendance(
        self,
        student_id: str,
        name: str,
        track_id: int,
        similarity: float = 0.0,
    ) -> Optional[AttendanceEvent]:
        """
        Mark a student present if they have not already
        been marked today.

        Returns:
            AttendanceEvent if attendance was newly marked.

            None if:
                - student is invalid
                - student was already marked today
        """

        # -------------------------------------------------
        # Validate student
        # -------------------------------------------------

        if not student_id:
            return None

        if not name:
            return None

        # -------------------------------------------------
        # Do not mark UNKNOWN
        # -------------------------------------------------

        if student_id.upper() == "UNKNOWN":
            return None

        # -------------------------------------------------
        # Check duplicate attendance
        # -------------------------------------------------

        if student_id in self.marked_students:

            print(
                f"[Attendance] Already marked: "
                f"{student_id} - {name}"
            )

            return None

        # -------------------------------------------------
        # Current date/time
        # -------------------------------------------------

        now = datetime.now()

        attendance_event = AttendanceEvent(
            student_id=student_id,
            name=name,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            status="Present",
            track_id=track_id,
            similarity=float(similarity),
        )

        # -------------------------------------------------
        # Store in memory
        # -------------------------------------------------

        self.marked_students[student_id] = attendance_event

        print(
            f"[Attendance] MARKED PRESENT: "
            f"{student_id} - {name} "
            f"at {attendance_event.time}"
        )

        return attendance_event

    # =====================================================
    # CHECK ATTENDANCE
    # =====================================================

    def is_marked(self, student_id: str) -> bool:
        """
        Returns True if the student has already been
        marked present.
        """

        return student_id in self.marked_students

    # =====================================================
    # GET ATTENDANCE
    # =====================================================

    def get_attendance(
        self,
        student_id: str,
    ) -> Optional[AttendanceEvent]:
        """
        Get the attendance event for one student.
        """

        return self.marked_students.get(student_id)

    # =====================================================
    # GET ALL ATTENDANCE
    # =====================================================

    def get_all_attendance(self) -> List[AttendanceEvent]:
        """
        Returns all attendance events created during
        this engine session.
        """

        return list(self.marked_students.values())

    # =====================================================
    # COUNT
    # =====================================================

    def attendance_count(self) -> int:
        """
        Returns the number of students marked present.
        """

        return len(self.marked_students)

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:
        """
        Clears the current in-memory attendance records.

        Useful for starting a new attendance session.
        """

        self.marked_students.clear()

        print(
            "[Attendance] Attendance session reset."
        )


# =========================================================
# SELF TEST
# =========================================================

def _run_self_test():

    print("========================================")
    print("Module 12 - Attendance Engine")
    print("========================================")

    engine = AttendanceEngine()

    # -----------------------------------------------------
    # Test 1
    # -----------------------------------------------------

    print("\n1. Marking first student...")

    event1 = engine.mark_attendance(
        student_id="STU008",
        name="somesh",
        track_id=1,
        similarity=0.72,
    )

    if event1 is not None:

        print(
            "PASS: Student marked present."
        )

        print(
            f"  Student ID: {event1.student_id}"
        )

        print(
            f"  Name:       {event1.name}"
        )

        print(
            f"  Date:       {event1.date}"
        )

        print(
            f"  Time:       {event1.time}"
        )

        print(
            f"  Status:     {event1.status}"
        )

        print(
            f"  Track ID:   {event1.track_id}"
        )

        print(
            f"  Similarity: {event1.similarity:.4f}"
        )

    else:

        print(
            "FAILED: Student was not marked."
        )

    # -----------------------------------------------------
    # Test 2
    # -----------------------------------------------------

    print("\n2. Testing duplicate attendance...")

    duplicate = engine.mark_attendance(
        student_id="STU008",
        name="somesh",
        track_id=1,
        similarity=0.75,
    )

    if duplicate is None:

        print(
            "PASS: Duplicate attendance prevented."
        )

    else:

        print(
            "FAILED: Duplicate attendance was created."
        )

    # -----------------------------------------------------
    # Test 3
    # -----------------------------------------------------

    print("\n3. Adding second student...")

    event2 = engine.mark_attendance(
        student_id="STU003",
        name="Mayur",
        track_id=2,
        similarity=0.81,
    )

    if event2 is not None:

        print(
            "PASS: Second student marked."
        )

    else:

        print(
            "FAILED: Second student was not marked."
        )

    # -----------------------------------------------------
    # Test 4
    # -----------------------------------------------------

    print("\n4. Testing UNKNOWN student...")

    unknown = engine.mark_attendance(
        student_id="UNKNOWN",
        name="Unknown",
        track_id=3,
        similarity=0.30,
    )

    if unknown is None:

        print(
            "PASS: Unknown student was not marked."
        )

    else:

        print(
            "FAILED: Unknown student was marked."
        )

    # -----------------------------------------------------
    # Test 5
    # -----------------------------------------------------

    print("\n5. Checking attendance count...")

    count = engine.attendance_count()

    print(
        f"Attendance count: {count}"
    )

    if count == 2:

        print(
            "PASS: Attendance count is correct."
        )

    else:

        print(
            "FAILED: Unexpected attendance count."
        )

    # -----------------------------------------------------
    # Test 6
    # -----------------------------------------------------

    print("\n6. Listing attendance...")

    for event in engine.get_all_attendance():

        print(
            f"  {event.student_id:<10}"
            f"{event.name:<15}"
            f"{event.date} "
            f"{event.time} "
            f"{event.status}"
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n========================================")
    print("Attendance Engine self-test complete")
    print("========================================")

    print(
        f"Students marked present: "
        f"{engine.attendance_count()}"
    )

    print(
        "All attendance engine tests passed."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    _run_self_test()