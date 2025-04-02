import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import xgboost as xgb




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

n = '8'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n

#os.sched_setaffinity(0, {0, 1, 2, 3}) 
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# Forcer la visibilité à 1 seul GPU

# Optionnel mais utile sur certains serveurs Linux
try:
    os.sched_setaffinity(0, set(range(8)))
except AttributeError:
    pass  # Si le système ne supporte pas (ex: Mac)



# -- CONFIG
CSV_PATH = '/home/miashs3/data/grp3/crops/raw_crops/labels.csv'         # Fichier d'entraînement
IMG_DIR = "/home/miashs3/data/grp3/crops/raw_crops/"
BATCH_SIZE = 32
NUM_EPOCHS = 30
IMG_SIZE = 224
NUM_CLASSES = 9
SEED = 42
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_WEIGHTS = torch.tensor([0.1] + [1.0]*8).to(DEVICE)
THRESHOLDS = np.array([0.7] + [0.5]*8)

def compute_f1(y_true, y_pred, exclude_autre=False):
    y_pred_bin = (y_pred > 0.5).astype(int)
    if exclude_autre:
        y_pred_bin = y_pred_bin[:, 1:]
        y_true = y_true[:, 1:]
    return f1_score(y_true, y_pred_bin, average='macro')

# -- TRANSFORMS
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class MultiLabelDataset(Dataset):
    def __init__(self, df, img_dir, transform, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['img_name'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.is_test:
            return image, os.path.splitext(row['img_name'])[0]
        else:
            label = torch.zeros(NUM_CLASSES)
            label[row['labels']] = 1.0
            return image, label

def get_model(features=False):
    model = models.resnet18(pretrained=True)
    if features:
        return nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    return model.to(DEVICE)

def train_cnn(df):
    dataset = MultiLabelDataset(df, IMG_DIR, transform=train_transform)
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    multilabels = np.zeros((len(df), NUM_CLASSES))
    for i, label_list in enumerate(df['labels']):
        multilabels[i, label_list] = 1
    train_idx, val_idx = next(mskf.split(df['img_name'], multilabels))
    train_ds = MultiLabelDataset(df.iloc[train_idx], IMG_DIR, train_transform)
    val_ds = MultiLabelDataset(df.iloc[val_idx], IMG_DIR, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model()
    criterion = nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHTS)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                outputs = torch.sigmoid(model(x)).cpu().numpy()
                all_preds.append(outputs)
                all_labels.append(y.numpy())

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        f1_total = compute_f1(y_true, y_pred)
        f1_no_autre = compute_f1(y_true, y_pred, exclude_autre=True)
        print(f"Epoch {epoch+1} - F1: {f1_total:.4f} | F1 sans AUTRE: {f1_no_autre:.4f}")
        if f1_total > best_f1:
            best_f1 = f1_total
            torch.save(model.state_dict(), "best_model_c.pth")
            print("✅ Meilleur modèle sauvegardé")

def extract_features(df):
    dataset = MultiLabelDataset(df, IMG_DIR, transform=val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = get_model(features=True)
    model.load_state_dict(torch.load("best_model_c.pth", map_location=DEVICE), strict=False)
    model.eval()

    X, y = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            out = model(images.to(DEVICE)).squeeze()
            X.append(out.cpu().numpy())
            y.extend(labels.argmax(dim=1).numpy())
    np.save("xgb_features.npy", np.vstack(X))
    np.save("xgb_labels.npy", np.array(y))

def train_xgb():
    X = np.load("xgb_features.npy")
    y = np.load("xgb_labels.npy")
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=NUM_CLASSES, nthread=8)
    model.fit(X, y)
    preds = model.predict(X)
    print("🎯 Accuracy XGB:", accuracy_score(y, preds))
    model.save_model("xgb_model.json")

def predict_xgb(img_dir):
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")
    model.set_params(nthread=8)

    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")])
    df_test = pd.DataFrame({'img_name': files})

    dataset = MultiLabelDataset(df_test, img_dir, val_transform, is_test=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    cnn = get_model(features=True)
    cnn.load_state_dict(torch.load("best_model.pth", map_location=DEVICE), strict=False)
    cnn.eval()

    X, ids = [], []
    with torch.no_grad():
        for images, image_ids in tqdm(loader):
            out = cnn(images.to(DEVICE)).squeeze()
            X.append(out.cpu().numpy())
            ids.extend(image_ids)

    X = np.vstack(X)
    preds = model.predict(X)

    pd.DataFrame({'idx': ids, 'gt': preds}).to_csv("submission_xgb.csv", index=False)
    print("✅ submission.csv généré")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'extract', 'xgb', 'predict_xgb', 'full'])
    parser.add_argument('--img_dir', type=str, help="dossier test images")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)

    def fusion_labels(row):
        return list(set([int(row[f'label{i}']) for i in range(1, 5)]))

    df['labels'] = df.apply(fusion_labels, axis=1)

    if args.mode == 'train':
        train_cnn(df)
    elif args.mode == 'extract':
        extract_features(df)
    elif args.mode == 'xgb':
        train_xgb()
    elif args.mode == 'predict_xgb':
        predict_xgb(args.img_dir)
    elif args.mode == 'full':
        train_cnn(df)
        extract_features(df)
        train_xgb()
        predict_xgb(args.img_dir)

if __name__ == "__main__":
    main()