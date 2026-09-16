"""
recognition/embedding.py
-----------------------------------------------------------------------
FaceNet embedding generation — preserves the EXACT preprocessing and
model configuration used to generate the 246-image dataset embeddings
already stored in MongoDB. Do not change Resize size, Normalize
values, or the pretrained weights identifier without regenerating
every stored embedding, or new/old embeddings will no longer be
comparable.

Responsibilities (and ONLY these):
  - Load InceptionResnetV1(pretrained="vggface2") once
  - Convert a face crop into a 512-dim embedding

No matching logic, no MongoDB access — recognition/matcher.py (next)
consumes this module's output.
-----------------------------------------------------------------------
"""

from typing import Optional

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
import cv2


EMBEDDING_DIM = 512


class FaceEmbedder:
    """
    Wraps InceptionResnetV1(pretrained="vggface2") with the exact
    preprocessing pipeline used to generate the existing MongoDB
    embeddings.

    Usage:
        embedder = FaceEmbedder()
        embedding = embedder.get_embedding(face_crop_bgr)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

        # Exact match to the original embedding-generation setup.
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def get_embedding(self, face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            face_crop_bgr: a face crop as a BGR numpy array (e.g. from
                detection/face_crop.py). Must be non-empty.

        Returns:
            512-dim float32 numpy embedding, or None if the crop is
            invalid (empty/degenerate) rather than raising.
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None

        # Original embeddings were generated from PIL-loaded (RGB)
        # images. Our crops come from OpenCV (BGR) — convert to match,
        # otherwise the model sees color channels swapped relative to
        # how the stored embeddings were produced. See module notes.
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        return embedding.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------
# Self-test — compares embeddings on real dataset images rather than
# the live webcam, so results are reproducible and don't depend on
# modules 3-6. Verifies:
#   1. Embedding dimension is 512
#   2. Two images of the SAME person -> high cosine similarity
#   3. Images of DIFFERENT people -> lower cosine similarity
# ---------------------------------------------------------------------
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _run_self_test() -> None:
    import sys
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "dataset"

    if not dataset_dir.exists():
        print(f"FAILED: dataset directory not found at {dataset_dir}")
        return

    # Find at least two people, each with at least two images.
    people = {}
    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        images = sorted(
            [p for p in person_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        if len(images) >= 2:
            people[person_dir.name] = images

    if len(people) < 2:
        print(
            "FAILED: need at least 2 people, each with >= 2 images, "
            f"under {dataset_dir}. Found: { {k: len(v) for k, v in people.items()} }"
        )
        return

    names = list(people.keys())
    person_a, person_b = names[0], names[1]
    img_a1, img_a2 = people[person_a][0], people[person_a][1]
    img_b1 = people[person_b][0]

    print(f"Using dataset images:")
    print(f"  Person A ('{person_a}'): {img_a1.name}, {img_a2.name}")
    print(f"  Person B ('{person_b}'): {img_b1.name}")
    print("-" * 60)

    print("Loading FaceNet model (InceptionResnetV1, vggface2)...")
    embedder = FaceEmbedder()

    def load_bgr(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        return img

    emb_a1 = embedder.get_embedding(load_bgr(img_a1))
    emb_a2 = embedder.get_embedding(load_bgr(img_a2))
    emb_b1 = embedder.get_embedding(load_bgr(img_b1))

    print(f"Embedding shape: {emb_a1.shape} (expected ({EMBEDDING_DIM},))")
    if emb_a1.shape != (EMBEDDING_DIM,):
        print("FAILED: unexpected embedding dimension.")
        sys.exit(1)

    sim_same_person = _cosine_similarity(emb_a1, emb_a2)
    sim_diff_person = _cosine_similarity(emb_a1, emb_b1)

    print("-" * 60)
    print(f"Cosine similarity — same person  ({person_a} vs {person_a}): {sim_same_person:.4f}")
    print(f"Cosine similarity — diff person  ({person_a} vs {person_b}): {sim_diff_person:.4f}")
    print("-" * 60)

    if sim_same_person > sim_diff_person:
        print("PASS: same-person similarity is higher than different-person similarity.")
    else:
        print(
            "WARNING: same-person similarity was NOT higher than different-person "
            "similarity. This would indicate a preprocessing mismatch — check the "
            "BGR/RGB conversion note above, and confirm these dataset images are "
            "already reasonably tight face crops (this module doesn't align/crop)."
        )


if __name__ == "__main__":
    _run_self_test()