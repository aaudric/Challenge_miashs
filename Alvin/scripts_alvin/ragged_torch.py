import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import Image
import pandas as pd
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RaggedDataset(Dataset):
    def __init__(self, batch_size=8, data_dir="/home/barrage/grp3/crops/", background_dir="/home/barrage/grp3/crops/background_patches/", max_background=300):
        self.batch_size = batch_size
        self.max_background = max_background
        self.backgrounds = []
        self.images = []
        self.labels = []
        self._load_background(background_dir)
        print("Background loaded")
        self._load_from_csv(data_dir, "raw_crops/labels.csv")
        print("Images loaded")
        print(len(self.images))

    def _load_background(self, folder):
        all_imgs = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        selected_imgs = random.sample(all_imgs, min(self.max_background, len(all_imgs)))
        
        for i in range(len(selected_imgs)):
            im = Image.open(os.path.join(folder, selected_imgs[i])).convert("RGB")
            im_np = np.array(im)
            h, w = im_np.shape[:2]
            new_h = h // 2
            new_w = w // 2
            im = im.resize((new_h, new_w))
            im_np = np.array(im, dtype=np.uint8)
            self.images.append(im_np)
            labels = np.zeros((9))
            labels[8] = 1
            self.labels.append(labels)

    def _load_from_csv(self, data_dir, csv_path):
        df = pd.read_csv(os.path.join(data_dir, csv_path))
        for _, row in df.iterrows():
            labels = [int(row["label1"]), int(row["label2"]), int(row["label3"]), int(row["label4"])]
            img_name = row["img_name"]
            img_path = os.path.join(data_dir, "raw_crops/"+img_name)
            label = np.zeros((9))
            for idx in labels:
                label[idx] = 1
            label /= np.sum(label)
            if os.path.exists(img_path):
                im = Image.open(img_path).convert("RGB")
                im_np = np.array(im)
                h, w = im_np.shape[:2]
                new_h = h // 2
                new_w = w // 2
                im = im.resize((new_h, new_w))
                im_np = np.array(im, dtype=np.uint8)
                self.images.append(im_np)
                la = np.zeros((9))
                la[np.argmax(label)] = 1
                self.labels.append(la)
            else :
                print("le lable n'existe pas")

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = (idx + 1) * self.batch_size

        batch_labels = torch.tensor(self.labels[start:stop], dtype=torch.float32)
        batch_images = []
        for i in range(start, stop):
            if i < len(self.images) :
                image = np.array(self.images[i], dtype=np.float32) / 255.0
                print(image.shape)
                image = torch.tensor(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                batch_images.append(image)
        return batch_images, batch_labels

    def augment_image(self, image):
        # Random flipping and brightness
        if random.random() > 0.5:
            image = np.fliplr(image)
        if random.random() > 0.5:
            image = np.flipud(image)
        image = np.clip(image + random.uniform(-0.1, 0.1), 0, 255)
        return image


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)
        x = self.conv6(x)
        x = torch.relu(x)
        x = self.conv7(x)
        x = torch.relu(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Training loop
batch_size = 32
dataset = RaggedDataset(batch_size=batch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = SimpleCNN().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for batch_images, batch_labels in dataloader:
    print(batch_images)  # Lot d'images
    print(batch_labels)  # Labels correspondants
    break

for epoch in range(10):
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, torch.max(y, 1)[1])  # CrossEntropy loss expects class indices
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), "model1.pth")
