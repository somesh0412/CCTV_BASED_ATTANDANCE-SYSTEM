"""
identity/identity_cache.py

Module 10: Identity Cache

Purpose:
    Remember which Student ID belongs to a DeepSORT Track ID.

Flow:

    Track ID
       ↓
    Identity Cache
       ↓
    Student ID + Name

Example:

    Track 1 → STU008 → somesh
    Track 2 → STU001 → Aditya

This module does NOT:
    - run YOLO
    - run DeepSORT
    - run FaceNet
    - perform face matching
    - access MongoDB
    - mark attendance
"""

from dataclasses import dataclass
from typing import Dict, Optional
import time


# ---------------------------------------------------------
# Cached identity
# ---------------------------------------------------------

@dataclass
class CachedIdentity:
    """
    Identity associated with one DeepSORT track.
    """

    track_id: int
    student_id: str
    name: str

    # Last time this track was seen.
    last_seen: float

    # Number of times this identity has been updated.
    update_count: int = 1


# ---------------------------------------------------------
# Identity Cache
# ---------------------------------------------------------

class IdentityCache:

    def __init__(
        self,
        expiry_seconds: float = 3.0,
    ):
        """
        Args:
            expiry_seconds:
                How long an unseen Track ID should remain
                in the cache.

        Example:

            Track 1 → STU008

            If Track 1 disappears for less than
            expiry_seconds, its cached identity remains.

            If it stays missing longer than that,
            the cache removes it.
        """

        self.expiry_seconds = expiry_seconds

        self._cache: Dict[
            int,
            CachedIdentity
        ] = {}

    # -----------------------------------------------------
    # Add / update identity
    # -----------------------------------------------------

    def set_identity(
        self,
        track_id: int,
        student_id: str,
        name: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Store or update the identity of a track.
        """

        if timestamp is None:
            timestamp = time.time()

        existing = self._cache.get(track_id)

        if existing is None:

            self._cache[track_id] = CachedIdentity(
                track_id=track_id,
                student_id=student_id,
                name=name,
                last_seen=timestamp,
                update_count=1,
            )

        else:

            existing.student_id = student_id
            existing.name = name
            existing.last_seen = timestamp
            existing.update_count += 1

    # -----------------------------------------------------
    # Get identity
    # -----------------------------------------------------

    def get_identity(
        self,
        track_id: int,
        timestamp: Optional[float] = None,
    ) -> Optional[CachedIdentity]:
        """
        Return the cached identity for a Track ID.

        Returns None if:
            - Track ID doesn't exist
            - cached identity has expired
        """

        if timestamp is None:
            timestamp = time.time()

        identity = self._cache.get(track_id)

        if identity is None:
            return None

        # Check expiry
        if (
            timestamp - identity.last_seen
            > self.expiry_seconds
        ):
            del self._cache[track_id]
            return None

        return identity

    # -----------------------------------------------------
    # Mark track as seen
    # -----------------------------------------------------

    def mark_seen(
        self,
        track_id: int,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Update last_seen time for an existing track.

        Returns:
            True  → track exists
            False → track not in cache
        """

        if timestamp is None:
            timestamp = time.time()

        identity = self._cache.get(track_id)

        if identity is None:
            return False

        identity.last_seen = timestamp

        return True

    # -----------------------------------------------------
    # Remove one track
    # -----------------------------------------------------

    def remove(
        self,
        track_id: int,
    ) -> None:
        """
        Remove a specific Track ID.
        """

        self._cache.pop(
            track_id,
            None,
        )

    # -----------------------------------------------------
    # Remove expired tracks
    # -----------------------------------------------------

    def cleanup(
        self,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Remove identities that have expired.

        Returns:
            Number of removed entries.
        """

        if timestamp is None:
            timestamp = time.time()

        expired_tracks = []

        for track_id, identity in self._cache.items():

            if (
                timestamp - identity.last_seen
                > self.expiry_seconds
            ):
                expired_tracks.append(
                    track_id
                )

        for track_id in expired_tracks:
            del self._cache[track_id]

        return len(expired_tracks)

    # -----------------------------------------------------
    # Check existence
    # -----------------------------------------------------

    def contains(
        self,
        track_id: int,
    ) -> bool:
        """
        Check whether a Track ID currently exists
        in the cache.
        """

        return track_id in self._cache

    # -----------------------------------------------------
    # Number of cached identities
    # -----------------------------------------------------

    def size(self) -> int:
        return len(self._cache)

    # -----------------------------------------------------
    # Clear cache
    # -----------------------------------------------------

    def clear(self) -> None:
        """
        Remove all cached identities.
        """

        self._cache.clear()

    # -----------------------------------------------------
    # Debug information
    # -----------------------------------------------------

    def get_all(self) -> Dict[
        int,
        CachedIdentity
    ]:
        """
        Return a copy of the current cache.
        """

        return dict(self._cache)


# ---------------------------------------------------------
# Self-test
# ---------------------------------------------------------

def run_self_test():

    print("========================================")
    print("Module 10 - Identity Cache")
    print("========================================")

    # Use a short expiry for testing.
    cache = IdentityCache(
        expiry_seconds=3.0
    )

    # Use controlled timestamps so the test
    # doesn't depend on actual waiting.
    start_time = 1000.0

    # -----------------------------------------------------
    # Test 1: Add identity
    # -----------------------------------------------------

    print("\n1. Adding identity...")

    cache.set_identity(
        track_id=1,
        student_id="STU008",
        name="somesh",
        timestamp=start_time,
    )

    identity = cache.get_identity(
        track_id=1,
        timestamp=start_time + 1,
    )

    if identity is None:
        print("FAILED: identity was not stored.")
        return

    print(
        f"Track {identity.track_id} → "
        f"{identity.student_id} → "
        f"{identity.name}"
    )

    # -----------------------------------------------------
    # Test 2: Retrieve cached identity
    # -----------------------------------------------------

    print("\n2. Testing cache retrieval...")

    identity = cache.get_identity(
        track_id=1,
        timestamp=start_time + 2,
    )

    if identity is None:
        print("FAILED: cached identity disappeared too early.")
        return

    print(
        "PASS: cached identity retrieved."
    )

    # -----------------------------------------------------
    # Test 3: Second student
    # -----------------------------------------------------

    print("\n3. Adding second track...")

    cache.set_identity(
        track_id=2,
        student_id="STU001",
        name="Aditya",
        timestamp=start_time,
    )

    print(
        f"Cached identities: {cache.size()}"
    )

    if cache.size() != 2:
        print(
            "FAILED: expected 2 cached identities."
        )
        return

    # -----------------------------------------------------
    # Test 4: Mark seen
    # -----------------------------------------------------

    print("\n4. Testing mark_seen...")

    success = cache.mark_seen(
        track_id=1,
        timestamp=start_time + 4,
    )

    if not success:
        print(
            "FAILED: mark_seen returned False."
        )
        return

    identity = cache.get_identity(
        track_id=1,
        timestamp=start_time + 6,
    )

    if identity is None:
        print(
            "FAILED: track expired despite mark_seen."
        )
        return

    print(
        "PASS: track activity refreshed."
    )

    # -----------------------------------------------------
    # Test 5: Expiry
    # -----------------------------------------------------

    print("\n5. Testing expiry...")

    identity = cache.get_identity(
        track_id=2,
        timestamp=start_time + 4,
    )

    if identity is not None:
        print(
            "FAILED: expired identity still exists."
        )
        return

    print(
        "PASS: expired identity removed."
    )

    # -----------------------------------------------------
    # Test 6: Cleanup
    # -----------------------------------------------------

    print("\n6. Testing cleanup...")

    removed = cache.cleanup(
        timestamp=start_time + 10
    )

    print(
        f"Expired identities removed: {removed}"
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n========================================")
    print("Identity Cache self-test complete")
    print("========================================")

    print(
        f"Current cache size: {cache.size()}"
    )

    print(
        "All identity cache tests passed."
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run_self_test()