import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from generator_torch import CustomDataset
#from torchmetrics.classification import Accuracy  # for calculating validation accuracy

# Set number of threads and GPU configuration
n = '3'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2})

# Load dataset
train_dataset = CustomDataset(path="../crops/croped_data.npz", transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Random brightness and contrast
]))

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)

# Validation data
validation_images, validation_labels = train_dataset.get_validation_data()
validation_images = validation_images.cuda()
validation_labels = validation_labels.cuda()

# Define model
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 9)  # 9 classes output
)
#model.load_state_dict(torch.load("./baseline_resnet.pth"))
# Move model to GPU
model = model.cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Accuracy metric
#val_acc = Accuracy(task="multiclass", num_classes=9).cuda()

# Training loop
for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print("LOSS = ", loss)

    # Validation step
    model.eval()
    with torch.no_grad():
        # Forward pass on validation set
        predictions = []
        labels = []
        for i in range(validation_images.shape[0]//16) :
            validation_predictions = model(validation_images[i:i+16])
            predictions.append(validation_predictions)
            labels.append(validation_labels[i:i+16])
        validation_predictions = torch.cat(predictions, axis=0)
        # Calculate accuracy
        validation_labels = torch.cat(labels, axis=0)
        _, predicted = torch.max(validation_predictions, 1)
        correct = (predicted == validation_labels).sum().item()
        total = validation_labels.size(0)
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    print("EPOCH ", epoch)
# Save model
torch.save(model.state_dict(), "baseline_resnet.pth")