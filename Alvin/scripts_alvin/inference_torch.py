import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import pandas as pd

# Setting device and environment
n = '3'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 9)  # 9 classes output
)
model.load_state_dict(torch.load("./baseline_resnet.pth"))
model.to(device)

# Prepare images
folder_path = "../datatest/"
img_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

ids = []
predictions = []
for img in img_files:

    # Read and preprocess image
    image = np.array(Image.open(folder_path + img).resize((224, 224), Image.NEAREST))
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW format
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Move image to device (GPU or CPU)
    image = image.to(device)

    # Prediction without gradient computation
    with torch.no_grad():
        prediction = torch.argmax(model(image), axis=-1)
        predictions.append(prediction.item())

    # Store the image id (filename without extension)
    print(img, img[:-4])
    ids.append(img[:-4])

# Print the results
print(ids, predictions)

df = pd.DataFrame({
    'idx': ids,
    'gt': predictions
})

print(df)
df.to_csv("predictions.csv", index=False)