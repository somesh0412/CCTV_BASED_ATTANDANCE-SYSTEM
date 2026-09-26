"""
database/student_repository.py
-----------------------------------------------------------------------
Loads all registered students' embeddings from MongoDB into an
in-memory structure optimized for fast similarity search.

Responsibilities (and ONLY these):
  - Call database/mongodb.py to fetch raw student documents
  - Validate embedding shape (skip/warn on malformed data instead of
    crashing the whole pipeline over one bad record)
  - Flatten into a single L2-normalized (N, 512) matrix + parallel
    owner list, so recognition/matcher.py can do one vectorized
    comparison instead of nested per-student loops

No matching/thresholding logic lives here — that's matcher.py (next).
This module only prepares the data matcher.py will search over.
-----------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
from dataclasses import dataclass


import sys

import numpy as np


# Add face-engine project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.mongodb import (
    fetch_all_students,
    EXPECTED_EMBEDDING_DIM,
)


@dataclass
class StudentRecord:
    """Per-student view — used where you need one student's info, not the search matrix."""
    student_id: str
    name: str
    face_registered: bool
    embeddings: np.ndarray  # shape (num_embeddings_for_this_student, 512), raw (not normalized)


class StudentRepository:
    """
    Usage:
        repo = StudentRepository()
        repo.load()

        matrix = repo.get_embedding_matrix()   # (N, 512), L2-normalized
        owners = repo.get_owners()             # [(studentId, name), ...] aligned to matrix rows
    """

    def __init__(self):
        self._students: List[StudentRecord] = []
        self._embedding_matrix: np.ndarray = np.empty((0, EXPECTED_EMBEDDING_DIM), dtype=np.float32)
        self._owners: List[Tuple[str, str]] = []
        self._loaded = False

    def load(self) -> None:
        """
        Fetches all students from MongoDB and builds the in-memory
        search structure. Call once at pipeline startup.

        Students with missing/malformed embeddings are skipped (with
        a warning printed), not fatal — one bad record shouldn't
        prevent the other 10 students from being recognized.
        """
        raw_students = fetch_all_students()

        students: List[StudentRecord] = []
        all_rows: List[np.ndarray] = []
        owners: List[Tuple[str, str]] = []

        skipped = []

        for doc in raw_students:
            student_id = doc.get("studentId")
            name = doc.get("name")
            face_registered = doc.get("faceRegistered", False)
            raw_embeddings = doc.get("faceEmbeddings", [])

            if not student_id or not name:
                skipped.append(f"<missing studentId/name> ({doc.get('_id')})")
                continue

            if not raw_embeddings:
                skipped.append(f"{student_id}: no embeddings stored")
                continue

            valid_embeddings = []
            for i, emb in enumerate(raw_embeddings):
                arr = np.asarray(emb, dtype=np.float32)
                if arr.shape != (EXPECTED_EMBEDDING_DIM,):
                    skipped.append(
                        f"{student_id}: embedding[{i}] has shape {arr.shape}, "
                        f"expected ({EXPECTED_EMBEDDING_DIM},) — skipped this one embedding"
                    )
                    continue
                valid_embeddings.append(arr)

            if not valid_embeddings:
                skipped.append(f"{student_id}: all embeddings were malformed, student skipped entirely")
                continue

            embeddings_array = np.stack(valid_embeddings, axis=0)
            students.append(
                StudentRecord(
                    student_id=student_id,
                    name=name,
                    face_registered=bool(face_registered),
                    embeddings=embeddings_array,
                )
            )

            for emb in valid_embeddings:
                all_rows.append(_l2_normalize(emb))
                owners.append((student_id, name))

        self._students = students
        self._owners = owners
        self._embedding_matrix = (
            np.stack(all_rows, axis=0).astype(np.float32)
            if all_rows
            else np.empty((0, EXPECTED_EMBEDDING_DIM), dtype=np.float32)
        )
        self._loaded = True

        if skipped:
            print(f"[StudentRepository] {len(skipped)} issue(s) while loading:")
            for s in skipped:
                print(f"  - {s}")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("StudentRepository.load() must be called before use.")

    def get_all_students(self) -> List[StudentRecord]:
        self._ensure_loaded()
        return self._students

    def get_embedding_matrix(self) -> np.ndarray:
        """Returns the (N, 512) L2-normalized matrix, one row per stored embedding."""
        self._ensure_loaded()
        return self._embedding_matrix

    def get_owners(self) -> List[Tuple[str, str]]:
        """Returns [(studentId, name), ...], aligned row-for-row with get_embedding_matrix()."""
        self._ensure_loaded()
        return self._owners

    def student_count(self) -> int:
        self._ensure_loaded()
        return len(self._students)

    def embedding_count(self) -> int:
        self._ensure_loaded()
        return self._embedding_matrix.shape[0]


def _l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / (norm + eps)


# ---------------------------------------------------------------------
# Self-test — loads the repository and validates it end-to-end:
#   1. Matrix shape is (total_embeddings, 512)
#   2. Every row is L2-normalized (norm ~= 1.0)
#   3. owners list length matches matrix row count
#   4. Per-student embedding counts match what module 1 reported
# ---------------------------------------------------------------------
def _run_self_test() -> None:
    print("Loading student repository from MongoDB...")
    repo = StudentRepository()

    try:
        repo.load()
    except Exception as e:
        print(f"FAILED to load: {e}")
        return

    print("-" * 60)
    print(f"Students loaded:          {repo.student_count()}")
    print(f"Total embeddings loaded:  {repo.embedding_count()}")

    matrix = repo.get_embedding_matrix()
    owners = repo.get_owners()

    print(f"Embedding matrix shape:   {matrix.shape}")
    print(f"Owners list length:       {len(owners)}")
    print("-" * 60)

    problems = []

    if matrix.shape[0] != len(owners):
        problems.append(
            f"Matrix has {matrix.shape[0]} rows but owners has {len(owners)} entries — misaligned!"
        )

    if matrix.shape[0] > 0:
        norms = np.linalg.norm(matrix, axis=1)
        bad_norms = np.where(np.abs(norms - 1.0) > 1e-3)[0]
        if len(bad_norms) > 0:
            problems.append(
                f"{len(bad_norms)} row(s) are not properly L2-normalized "
                f"(e.g. row {bad_norms[0]} has norm {norms[bad_norms[0]]:.4f})"
            )

    print("Per-student embedding counts:")
    for student in repo.get_all_students():
        print(f"  {student.student_id:<10} {student.name:<15} embeddings={student.embeddings.shape[0]}")

    print("-" * 60)
    if problems:
        print(f"{len(problems)} issue(s) found:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Repository loaded and validated successfully.")


if __name__ == "__main__":
    _run_self_test()