import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, path="../crops/croped_data.npz", transform=None):
        data = np.load(path, allow_pickle=True)
        self.images = data["images"] / 255.0
        labels = data["labels"]
        labels = labels.item()
        
        self.labels = []
        for i in range(self.images.shape[0]):
            counter = np.zeros((9))
            counter[int(labels["label1"][i])] += 1
            counter[int(labels["label2"][i])] += 1
            counter[int(labels["label3"][i])] += 1
            counter[int(labels["label4"][i])] += 1

            classe = np.argmax(counter)
            self.labels.append(classe)
        self.labels = np.array(self.labels)

        indices = np.arange(len(self.images))
        random.shuffle(indices)
        cut = int(0.8 * self.images.shape[0])
        self.train_images = self.images[indices[:cut]]
        self.train_labels = self.labels[indices[:cut]]
        self.validation_images = self.images[indices[cut:]]
        self.validation_labels = self.labels[indices[cut:]]
        
        self.transform = transform

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        image = self.train_images[idx]
        label = self.train_labels[idx]
        
        image = np.transpose(image, (2, 0, 1))  # Convert from (H, W, C) to (C, H, W)
        image = torch.tensor(image, dtype=torch.float32)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_validation_data(self):
        # Convert validation data into a dataset for evaluation
        validation_images = np.transpose(self.validation_images, (0, 3, 1, 2))  # (N, C, H, W)
        validation_images = torch.tensor(validation_images, dtype=torch.float32)
        validation_labels = torch.tensor(self.validation_labels, dtype=torch.long)

        return validation_images, validation_labels


# Define transformations for data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Random brightness and contrast
])


train_dataset = CustomDataset(path="../crops/croped_data.npz", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_images, validation_labels = train_dataset.get_validation_data()
