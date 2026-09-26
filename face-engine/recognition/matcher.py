"""
recognition/matcher.py

Module 9: Face Matching

Flow:

    Face Crop
        ↓
    FaceNet Embedding
        ↓
    512-D embedding
        ↓
    StudentRepository
        ↓
    246 stored embeddings
        ↓
    Cosine similarity
        ↓
    Best matching student

This module does NOT:
- access MongoDB directly
- run YOLO
- run DeepSORT
- run FaceNet itself
- manage Track IDs
- mark attendance

The FaceEmbedder generates the live embedding.
The StudentRepository provides the stored embeddings.
This module only performs matching.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import sys

import numpy as np


# Add face-engine project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from recognition.embedding import (
    FaceEmbedder,
    EMBEDDING_DIM,
)

from database.student_repository import (
    StudentRepository,
)


# ---------------------------------------------------------
# Match result
# ---------------------------------------------------------

from dataclasses import dataclass


@dataclass
class MatchResult:
    """
    Result returned by the matcher.

    student_id:
        Registered student ID.

    name:
        Registered student name.

    similarity:
        Cosine similarity with the closest stored embedding.

    embedding_index:
        Row number of the closest embedding in the repository.

    matched:
        Whether the result passed the supplied threshold.
    """

    student_id: str
    name: str
    similarity: float
    embedding_index: int
    matched: bool


# ---------------------------------------------------------
# Face Matcher
# ---------------------------------------------------------

class FaceMatcher:

    def __init__(
        self,
        repository: StudentRepository,
        threshold: float = 0.40,
    ):
        """
        Args:
            repository:
                Loaded StudentRepository.

            threshold:
                Initial cosine similarity threshold.

        IMPORTANT:
            0.40 is only a STARTING VALUE.
            We will tune this after testing real images.
        """

        self.repository = repository
        self.threshold = threshold

        self.embedding_matrix = (
            repository.get_embedding_matrix()
        )

        self.owners = repository.get_owners()

        if self.embedding_matrix.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected embeddings with dimension "
                f"{EMBEDDING_DIM}, but received "
                f"{self.embedding_matrix.shape}"
            )

        if len(self.owners) != self.embedding_matrix.shape[0]:
            raise ValueError(
                "Embedding matrix and owners list "
                "are not aligned."
            )

    # -----------------------------------------------------
    # Normalize vector
    # -----------------------------------------------------

    @staticmethod
    def _normalize(
        embedding: np.ndarray,
    ) -> np.ndarray:

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(embedding)

        if norm < 1e-10:
            raise ValueError(
                "Cannot normalize a zero-length embedding."
            )

        return embedding / norm

    # -----------------------------------------------------
    # Match embedding
    # -----------------------------------------------------

    def match(
        self,
        embedding: np.ndarray,
    ) -> Optional[MatchResult]:
        """
        Compare one live FaceNet embedding against
        all stored student embeddings.

        Returns:
            MatchResult for the best candidate.

        Returns None if:
            - no embeddings are loaded
            - embedding has incorrect shape
            - embedding is invalid
        """

        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Expected embedding shape "
                f"({EMBEDDING_DIM},), "
                f"received {embedding.shape}"
            )

        # Normalize live embedding
        query = self._normalize(embedding)

        if self.embedding_matrix.shape[0] == 0:
            return None

        # -------------------------------------------------
        # Cosine similarity
        #
        # Repository embeddings are already normalized.
        # Therefore:
        #
        # cosine similarity = dot product
        # -------------------------------------------------

        similarities = (
            self.embedding_matrix @ query
        )

        # Find highest similarity
        best_index = int(
            np.argmax(similarities)
        )

        best_similarity = float(
            similarities[best_index]
        )

        student_id, name = (
            self.owners[best_index]
        )

        matched = (
            best_similarity >= self.threshold
        )

        return MatchResult(
            student_id=student_id,
            name=name,
            similarity=best_similarity,
            embedding_index=best_index,
            matched=matched,
        )


# ---------------------------------------------------------
# Self-test
# ---------------------------------------------------------

def run_self_test():

    import cv2
    from pathlib import Path

    print("========================================")
    print("Module 9 - Face Matcher")
    print("========================================")

    # -----------------------------------------------------
    # Load repository
    # -----------------------------------------------------

    print("\nLoading student repository...")

    repository = StudentRepository()
    repository.load()

    print(
        f"Students: {repository.student_count()}"
    )

    print(
        f"Embeddings: {repository.embedding_count()}"
    )

    # -----------------------------------------------------
    # Load FaceNet
    # -----------------------------------------------------

    print(
        "\nLoading FaceNet embedding model..."
    )

    embedder = FaceEmbedder()

    print(
        "FaceNet loaded successfully."
    )

    # -----------------------------------------------------
    # Create matcher
    # -----------------------------------------------------

    # 0.40 is only an initial testing threshold.
    matcher = FaceMatcher(
        repository,
        threshold=0.40,
    )

    print(
        "Matcher initialized."
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    dataset_dir = (
        project_root / "dataset"
    )

    if not dataset_dir.exists():
        print(
            f"\nFAILED: dataset not found at "
            f"{dataset_dir}"
        )
        return

    # -----------------------------------------------------
    # Test several students
    # -----------------------------------------------------

    print("\n========================================")
    print("Testing known student images")
    print("========================================")

    test_count = 0
    correct_count = 0

    for student_dir in sorted(
        dataset_dir.iterdir()
    ):

        if not student_dir.is_dir():
            continue

        images = sorted(
            [
                path
                for path in student_dir.iterdir()
                if path.suffix.lower()
                in (".jpg", ".jpeg", ".png")
            ]
        )

        if not images:
            continue

        # Use first image for this student
        image_path = images[0]

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"Could not read {image_path}"
            )
            continue

        # Generate embedding
        embedding = (
            embedder.get_embedding(image)
        )

        if embedding is None:
            print(
                f"Could not generate embedding "
                f"for {image_path}"
            )
            continue

        # Match
        result = matcher.match(
            embedding
        )

        if result is None:
            print(
                f"No match returned for "
                f"{student_dir.name}"
            )
            continue

        expected_name = student_dir.name

        is_correct = (
            result.name.lower()
            == expected_name.lower()
        )

        if is_correct:
            correct_count += 1

        test_count += 1

        status = (
            "CORRECT"
            if is_correct
            else "WRONG"
        )

        print(
            f"{expected_name:<15} → "
            f"{result.student_id:<8} "
            f"{result.name:<15} "
            f"similarity={result.similarity:.4f} "
            f"{status}"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n========================================")
    print("Matcher self-test complete")
    print("========================================")

    print(
        f"Students tested: {test_count}"
    )

    print(
        f"Correct matches: {correct_count}"
    )

    if test_count > 0:

        accuracy = (
            correct_count
            / test_count
            * 100
        )

        print(
            f"Accuracy:        {accuracy:.1f}%"
        )

    print(
        "\nNOTE:"
    )

    print(
        "The threshold 0.40 is only a starting "
        "value. Do not treat it as the final "
        "recognition threshold yet."
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run_self_test()