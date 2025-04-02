import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ----------- ENV SETUP -----------
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = '8'
os.environ["OPENBLAS_NUM_THREADS"] = '8'
os.environ["MKL_NUM_THREADS"] = '8'
os.environ["NUMEXPR_NUM_THREADS"] = '8'
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
try:
    os.sched_setaffinity(0, set(range(8)))
except:
    pass

# ----------- CONFIG -----------
CSV_PATH = "/home/miashs3/data/grp3/crops/raw_crops/labels.csv"            # Fichier d'entraînement
IMG_DIR = "/home/miashs3/data/grp3/crops/raw_crops/"
PSEUDO_LABELS_PATH = '/home/miashs3/pseudo_labels.csv'
BATCH_SIZE = 32
NUM_EPOCHS = 30
IMG_SIZE = 224
NUM_CLASSES = 9
SEED = 42
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_WEIGHTS = torch.tensor([0.1] + [1.0]*8).to(DEVICE)

# ----------- TRANSFORMS -----------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------- DATASETS -----------
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['img_name'])).convert("RGB")
        img = self.transform(img)
        return img, int(row['label'])

class TestDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        img = self.transform(img)
        return img, fname

# ----------- MODEL & F1 -----------
def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

# ----------- LABEL FUSION -----------
def prepare_training_data(csv_path):
    df = pd.read_csv(csv_path)
    def fuse_labels(row):
        labels = [int(row[f'label{i}']) for i in range(1, 5)]
        return Counter(labels).most_common(1)[0][0]
    df['label'] = df.apply(fuse_labels, axis=1)
    return df[['img_name', 'label']]

# ----------- TRAINING -----------
def train_model(df, img_dir, out_model="model_r.pth"):
    train_df, val_df = train_test_split(df, stratify=df['label'], test_size=0.2, random_state=SEED)
    train_ds = ImageDataset(train_df, img_dir, train_transform)
    val_ds = ImageDataset(val_df, img_dir, test_transform)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)

    model = get_model()
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    best_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                preds = model(x).argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y.numpy())

        f1 = compute_f1(y_true, y_pred)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out_model)
            print(f"✅ Meilleur modèle sauvegardé (F1: {f1:.4f})")

# ----------- PSEUDO-LABELING -----------
def generate_pseudo_labels(img_dir, model_path="model_r.pth", threshold=0.9):
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = TestDataset(img_dir, test_transform)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)

    results = []
    with torch.no_grad():
        for images, fnames in tqdm(loader, desc="Pseudo-labelling"):
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1)
            confs, preds = torch.max(probs, dim=1)

            for fname, conf, pred in zip(fnames, confs.cpu(), preds.cpu()):
                if conf.item() >= threshold:
                    results.append({"img_name": fname, "label": pred.item()})

    pd.DataFrame(results).to_csv(PSEUDO_LABELS_PATH, index=False)
    print(f"✅ Pseudo-labels générés : {len(results)} → {PSEUDO_LABELS_PATH}")

# ----------- RE-TRAIN -----------
def retrain_with_pseudo(train_img_dir):
    real_df = prepare_training_data(CSV_PATH)
    pseudo_df = pd.read_csv(PSEUDO_LABELS_PATH)
    combined_df = pd.concat([real_df, pseudo_df], ignore_index=True)
    print(f"🔁 Réentraînement avec {len(combined_df)} images")
    train_model(combined_df, img_dir=train_img_dir, out_model="model_retrained.pth")

# ----------- INFERENCE -----------
def inference(img_dir, model_path="model_retrained.pth", out_csv="submission_pseudo_labels.csv"):
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = TestDataset(img_dir, test_transform)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)

    results = []
    with torch.no_grad():
        for images, files in tqdm(loader, desc="Inférence"):
            images = images.to(DEVICE)
            preds = model(images).argmax(dim=1).cpu().numpy()
            for fname, pred in zip(files, preds):
                results.append({"idx": os.path.splitext(fname)[0], "gt": pred})

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✅ Soumission sauvegardée dans {out_csv}")

# ----------- MAIN -----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'pseudo', 'retrain', 'infer', 'full'], required=True)
    parser.add_argument('--train_img_dir', type=str, required=True, help="Dossier des images d'entraînement")
    parser.add_argument('--img_dir', type=str, required=True, help="Dossier des images test")
    args = parser.parse_args()

    if args.mode == "train":
        df = prepare_training_data(CSV_PATH)
        train_model(df, img_dir=args.train_img_dir)

    elif args.mode == "pseudo":
        generate_pseudo_labels(args.img_dir)

    elif args.mode == "retrain":
        retrain_with_pseudo(args.train_img_dir)

    elif args.mode == "infer":
        inference(args.img_dir)

    elif args.mode == "full":
        df = prepare_training_data(CSV_PATH)
        train_model(df, img_dir=args.train_img_dir)
        generate_pseudo_labels(args.img_dir)
        retrain_with_pseudo(args.train_img_dir)
        inference(args.img_dir)

if __name__ == "__main__":
    main()