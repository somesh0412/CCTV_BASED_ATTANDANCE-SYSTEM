
import pickle
import numpy as np

with open("embeddings.pkl", "rb") as file:
    data = pickle.load(file)

for student, embeddings in data.items():
    print("\nStudent:", student)
    print("Number of stored items:", len(embeddings))
    print("First item type:", type(embeddings[0]))

    first = np.asarray(embeddings[0])

    print("First item shape:", first.shape)
    print("First item data type:", first.dtype)
    print("First item sample:", first.flatten()[:5])

    break
