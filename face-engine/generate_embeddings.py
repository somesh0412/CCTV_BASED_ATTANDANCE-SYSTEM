import os
import glob
import pickle

import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1


# --------------------------------------------------
# 1. Load FaceNet model
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


# --------------------------------------------------
# 3. Dataset location
# --------------------------------------------------

dataset_path = "dataset"

embeddings_data = {}

total_images = 0
successful_images = 0


# --------------------------------------------------
# 4. Process every student's folder
# --------------------------------------------------

for student_name in os.listdir(dataset_path):

    student_folder = os.path.join(dataset_path, student_name)

    if not os.path.isdir(student_folder):
        continue

    print(f"\nProcessing student: {student_name}")

    student_embeddings = []

    image_files = glob.glob(
        os.path.join(student_folder, "*")
    )

    for image_path in image_files:

        try:
            image = Image.open(image_path).convert("RGB")

            # Preprocess image
            face_tensor = transform(image)

            # Add batch dimension
            face_tensor = face_tensor.unsqueeze(0).to(device)

            # Generate embedding
            with torch.no_grad():
                embedding = model(face_tensor)

            # Convert tensor to Python list
            embedding = embedding.cpu().numpy()[0].tolist()

            student_embeddings.append({
                "image": os.path.basename(image_path),
                "embedding": embedding
            })

            successful_images += 1

            print(
                f"  ✓ {os.path.basename(image_path)} "
                f"→ embedding generated"
            )

        except Exception as error:

            print(
                f"  ✗ {os.path.basename(image_path)} "
                f"→ ERROR: {error}"
            )

        total_images += 1

    embeddings_data[student_name] = student_embeddings


# --------------------------------------------------
# 5. Save embeddings locally
# --------------------------------------------------

output_file = "embeddings.pkl"

with open(output_file, "wb") as file:
    pickle.dump(embeddings_data, file)


# --------------------------------------------------
# 6. Summary
# --------------------------------------------------

print("\n====================================")
print("Embedding generation completed")
print("====================================")

print(f"Total images:      {total_images}")
print(f"Successful:        {successful_images}")
print(f"Failed:            {total_images - successful_images}")

print(f"\nSaved to: {output_file}")