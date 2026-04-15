import os
import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet

embedder = FaceNet()

data = {}
dataset_path = "faces"
print("Program started")
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    
    embeddings = []
    
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        img = cv2.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        
        emb = embedder.embeddings(img)[0]
        embeddings.append(emb)
    
    if embeddings:
        data[person] = embeddings

with open("face_data.pkl", "wb") as f:
    pickle.dump(data, f)

print("Pickle file created ✅")