
import pickle
import numpy as np

with open("embeddings.pkl", "rb") as file:
    data = pickle.load(file)

print("Data type:", type(data))

if isinstance(data, dict):
    print("Keys:", list(data.keys())[:10])

    for key, value in list(data.items())[:3]:
        print("\nStudent:", key)
        print("Value type:", type(value))

        if isinstance(value, dict):
            print("Student fields:", list(value.keys()))

        try:
            arr = np.asarray(value)
            print("Embedding shape:", arr.shape)
        except Exception as e:
            print("Cannot convert:", e)

elif isinstance(data, (list, tuple)):
    print("Number of records:", len(data))
    if len(data) > 0:
        print("First record type:", type(data[0]))
        print("First record:", str(data[0])[:500])
