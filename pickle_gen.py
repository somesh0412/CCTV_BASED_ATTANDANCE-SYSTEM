"""
pickle_gen.py - Face Embedding Generator
==========================================
Reads all images from faces/<PersonName>/ and generates 128-d FaceNet
embeddings. Saves them to face_data.pkl.

Uses keras-facenet (TensorFlow backend).
Requires Python 3.10 or 3.11.

Usage: python pickle_gen.py
"""

import os
import cv2
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_facenet import FaceNet

FACES_DIR  = "faces"
OUTPUT_PKL = "face_data.pkl"
IMG_SIZE   = (160, 160)   # FaceNet input size


def main():
    print("=" * 52)
    print("   Face Embedding Generator  (FaceNet)")
    print("=" * 52)

    if not os.path.exists(FACES_DIR):
        print(f"\n❌  '{FACES_DIR}' folder not found.")
        print("    Run capture.py first to collect face images.")
        return

    print("\nLoading FaceNet model (first run may download weights)...")
    embedder = FaceNet()
    print("FaceNet loaded ✅\n")

    data = {}
    total_ok = 0
    total_skip = 0

    for person_name in sorted(os.listdir(FACES_DIR)):
        person_path = os.path.join(FACES_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        img_files = [f for f in os.listdir(person_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not img_files:
            print(f"  ⚠  No images for '{person_name}', skipping.")
            continue

        print(f"Processing: {person_name}  ({len(img_files)} images)")
        embeddings = []

        for img_name in img_files:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"    ⚠  Cannot read {img_name}, skipping.")
                total_skip += 1
                continue

            # FaceNet needs RGB + 160x160
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, IMG_SIZE)
            img_batch = np.expand_dims(img_resized, axis=0)

            emb = embedder.embeddings(img_batch)[0]
            embeddings.append(emb)
            total_ok += 1

        if embeddings:
            data[person_name] = embeddings
            print(f"  ✅  {len(embeddings)} embeddings saved")
        else:
            print(f"  ✗   No valid embeddings for '{person_name}'")

    if not data:
        print("\n❌  No data to save.")
        return

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(data, f)

    print(f"\n{'=' * 52}")
    print(f"✅  Saved: {OUTPUT_PKL}")
    print(f"   Persons   : {len(data)}")
    print(f"   Embeddings: {total_ok}")
    if total_skip:
        print(f"   Skipped   : {total_skip}")
    print(f"{'=' * 52}")
    print("\n→  Now run: python app.py")


if __name__ == "__main__":
    main()
