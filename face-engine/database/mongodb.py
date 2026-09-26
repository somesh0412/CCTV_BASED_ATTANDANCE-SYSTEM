"""
database/mongodb.py
-----------------------------------------------------------------------
Single point of contact between the Face Engine and MongoDB.

Responsibilities (and ONLY these):
  - Load MONGO_URI from .env
  - Connect to database "test", collection "students"
  - Expose fetch_all_students() for the rest of the pipeline

No embedding math, no matching logic, no caching — this file only
knows how to get raw student documents out of MongoDB.
-----------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, ConfigurationError

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
# Resolve .env relative to this file (face-engine/.env), not the
# current working directory — so this works no matter where you run
# the script from.
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "test")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "students")

# How long to wait for the server before giving up (ms). Fails fast
# instead of hanging if the URI/network is wrong.
SERVER_SELECTION_TIMEOUT_MS = 5000

EXPECTED_EMBEDDING_DIM = 512  # InceptionResnetV1(vggface2) output size


# ---------------------------------------------------------------------
# Connection (lazy singleton — created on first use, reused after)
# ---------------------------------------------------------------------
_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    """Returns a shared MongoClient, creating it on first call."""
    global _client

    if not MONGO_URI:
        raise RuntimeError(
            f"MONGO_URI is not set. Expected it in {ENV_PATH} — "
            "check that the .env file exists and contains MONGO_URI=..."
        )

    if _client is None:
        _client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
        )

    return _client


def get_students_collection():
    """Returns the test.students collection handle."""
    client = get_client()
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


def ping() -> bool:
    """
    Cheap connectivity check. Raises on failure rather than returning
    False, so callers get a clear error instead of a silent no-op.
    """
    client = get_client()
    client.admin.command("ping")
    return True


# ---------------------------------------------------------------------
# Public API used by the rest of the Face Engine
# ---------------------------------------------------------------------
def fetch_all_students() -> List[Dict[str, Any]]:
    """
    Fetches every student document from test.students, unmodified.

    Returns:
        List of dicts, each shaped like:
        {
          "studentId": str,
          "name": str,
          "faceRegistered": bool,
          "faceEmbeddings": [[float, ...512], ...]
        }
    """
    collection = get_students_collection()
    return list(collection.find({}))


def get_student_count() -> int:
    """Fast count without pulling embedding data across the wire."""
    collection = get_students_collection()
    return collection.count_documents({})


# ---------------------------------------------------------------------
# Self-test — run this file directly to verify the connection and
# validate the shape of what's already stored, without printing any
# sensitive data (no URI, no raw embedding values).
# ---------------------------------------------------------------------
def _run_self_test() -> None:
    print(f"Loading .env from: {ENV_PATH}")
    print(f"Target DB:         {DB_NAME}")
    print(f"Target collection: {COLLECTION_NAME}")
    print("-" * 60)

    try:
        ping()
        print("Connection OK (ping succeeded)\n")
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print("FAILED to connect to MongoDB.")
        print(f"Reason: {e}")
        print(
            "\nCheck: is MONGO_URI set correctly in .env, is your IP "
            "allow-listed in MongoDB Atlas, and is your network up?"
        )
        sys.exit(1)
    except (RuntimeError, ConfigurationError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    try:
        students = fetch_all_students()
    except Exception as e:
        print(f"Connected, but failed to fetch students: {e}")
        sys.exit(1)

    print(f"Total students found: {len(students)}")
    print("-" * 60)

    problems = []

    for student in students:
        student_id = student.get("studentId", "<MISSING studentId>")
        name = student.get("name", "<MISSING name>")
        registered = student.get("faceRegistered", None)
        embeddings = student.get("faceEmbeddings", [])

        num_embeddings = len(embeddings)
        first_dim = len(embeddings[0]) if num_embeddings > 0 else 0

        status = "OK"
        if num_embeddings == 0:
            status = "WARNING: no embeddings"
            problems.append(f"{student_id}: no embeddings stored")
        elif first_dim != EXPECTED_EMBEDDING_DIM:
            status = f"WARNING: dim={first_dim}, expected {EXPECTED_EMBEDDING_DIM}"
            problems.append(f"{student_id}: embedding dim {first_dim} != {EXPECTED_EMBEDDING_DIM}")

        print(
            f"  {student_id:<10} {name:<15} "
            f"faceRegistered={str(registered):<5} "
            f"embeddings={num_embeddings:<3} "
            f"dim={first_dim:<4} [{status}]"
        )

    print("-" * 60)
    if problems:
        print(f"{len(problems)} issue(s) found:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("All student records look structurally valid.")


if __name__ == "__main__":
    _run_self_test()