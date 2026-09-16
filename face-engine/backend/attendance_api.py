"""
backend/attendance_api.py

Module 15: Python -> Node.js automatic attendance client
"""

from typing import Optional

import requests


class AttendanceAPI:

    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        endpoint: str = "/api/attendance",
        timeout: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.timeout = timeout

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint}"

    def send_event(self, event) -> bool:
        """
        Sends an AttendanceEvent created by AttendanceEngine
        to the Node.js backend.

        Returns:
            True  -> request accepted
            False -> request failed
        """

        payload = {
            "studentId": event.student_id,
            "name": event.name,
            "date": event.date,
            "time": event.time,
            "status": event.status,
            "trackId": event.track_id,
            "similarity": float(event.similarity),
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
            )

        except requests.RequestException as error:

            print(
                f"[AttendanceAPI] "
                f"Connection failed: {error}"
            )

            return False

        # -------------------------------------------------
        # New attendance created
        # -------------------------------------------------

        if response.status_code == 201:

            print(
                f"[AttendanceAPI] "
                f"Attendance saved: "
                f"{event.student_id} - {event.name}"
            )

            return True

        # -------------------------------------------------
        # Already marked
        # -------------------------------------------------

        if response.status_code == 200:

            try:
                data = response.json()

                if data.get("alreadyMarked"):

                    print(
                        f"[AttendanceAPI] "
                        f"Already recorded: "
                        f"{event.student_id} - {event.name}"
                    )

                    return True

            except ValueError:
                pass

        # -------------------------------------------------
        # Other backend error
        # -------------------------------------------------

        print(
            f"[AttendanceAPI] "
            f"Backend returned HTTP "
            f"{response.status_code}"
        )

        try:
            print(
                f"[AttendanceAPI] "
                f"Response: {response.json()}"
            )

        except ValueError:
            print(
                f"[AttendanceAPI] "
                f"Response: {response.text}"
            )

        return False


# =========================================================
# SELF TEST
# =========================================================

def _run_self_test():

    print("========================================")
    print("Module 15 - Attendance API Client")
    print("========================================")

    api = AttendanceAPI()

    print("\nAPI URL:")
    print(api.url)

    print("\nAttendance API client initialized.")
    print("Ready to send AttendanceEvent objects.")

    print("\n========================================")
    print("Module 15 self-test complete")
    print("========================================")


if __name__ == "__main__":
    _run_self_test()