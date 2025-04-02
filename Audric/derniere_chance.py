import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score

# ----- CONFIGURATION -----
CSV_PATH = "/home/miashs3/data/grp3/crops/raw_crops/labels.csv"  # Fichier d'entraînement
IMG_DIR = "/home/miashs3/data/grp3/crops/raw_crops/"
BATCH_SIZE = 32
NUM_EPOCHS = 30
IMG_SIZE = 224
NUM_CLASSES = 9  # 9 classes au total (la classe 3 sera ignorée)
SEED = 42
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pour le multi-threading et la gestion CPU
n = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3})

# ----- FONCTION POUR AJOUTER DU CONTEXTE -----
def add_context(img, factor=1.4):
    """
    Agrandit l'image par un facteur donné en ajoutant des marges
    avec remplissage par réflexion pour conserver du contexte.
    """
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    pad_w, pad_h = new_w - w, new_h - h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return transforms.functional.pad(img, (left, top, right, bottom), padding_mode='reflect')

# ----- TRANSFORMS -----
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: add_context(img, factor=1.4)),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: add_context(img, factor=1.4)),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ----- DATASET -----
class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, is_test=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['img_name'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            label = torch.zeros(NUM_CLASSES)  # label factice pour l'inférence
        else:
            label_indices = row['labels']  # liste d'indices (incluant potentiellement 3)
            label = torch.zeros(NUM_CLASSES)
            label[label_indices] = 1.0

        return image, label

# ----- MODÈLE -----
def get_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    return model

# ----- MÉTRIQUES -----
def compute_f1(y_true, y_pred, exclude_autre=False):
    y_pred_bin = (y_pred > 0.5).astype(int)
    if exclude_autre:
        y_pred_bin = y_pred_bin[:, 1:]
        y_true = y_true[:, 1:]
    return f1_score(y_true, y_pred_bin, average='macro')

# ----- SUBMISSION -----
def generate_submission(model, full_df, transform, output_csv="submission.csv"):
    dataset = MultiLabelDataset(full_df, IMG_DIR, transform=transform, is_test=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds_bin = []
    filenames = full_df['img_name'].tolist()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            # On définit les seuils par classe
            thresholds = np.array([0.7] + [0.5] * (NUM_CLASSES - 1))
            # Calcul binaire
            preds = (probs > thresholds).astype(int)
            # On force la classe 3 à être ignorée
            preds[:, 3] = 0
            preds_bin.extend(preds)

    ids = [os.path.splitext(name)[0] for name in filenames]
    gt = ['[' + ', '.join(str(i) for i in np.where(p == 1)[0]) + ']' for p in preds_bin]

    df_out = pd.DataFrame({'idx': ids, 'gt': gt})
    df_out.to_csv(output_csv, index=False)
    print(f"📁 Fichier de soumission généré : {output_csv}")

# ----- MAIN -----
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(CSV_PATH)

    # Fusionner les labels des 4 experts en un seul vecteur multilabel par image
    # Ici, on ne supprime pas la classe 3 dans les données d'entraînement,
    # mais nous allons l'ignorer dans la fonction de perte.
    def fusion_labels(row):
        labels = [int(row[f'label{i}']) for i in range(1, 5)]
        return list(set(labels))  # ne retire pas la classe 3

    df['labels'] = df.apply(fusion_labels, axis=1)

    # Reconstruction du tableau multilabel pour la stratification
    multilabels = np.zeros((len(df), NUM_CLASSES))
    for i, label_list in enumerate(df['labels']):
        multilabels[i, label_list] = 1

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    train_idx, val_idx = next(mskf.split(df['img_name'], multilabels))
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]

    train_dataset = MultiLabelDataset(df_train, IMG_DIR, transform=train_transform)
    val_dataset = MultiLabelDataset(df_val, IMG_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model().to(DEVICE)
    # Les poids de classe (exemple : la classe 0 "AUTRE" est sous-pondérée)
    CLASS_WEIGHTS = torch.tensor([0.1] + [1.0]*8).to(DEVICE)
    # Utilisation d'une loss avec reduction='none' pour pouvoir masquer la classe 3
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            # Calcul de la loss par élément
            loss_tensor = criterion(outputs, labels)
            # Masque : on ignore la classe 3 (index 3)
            mask = torch.ones_like(loss_tensor)
            mask[:, 3] = 0.0
            loss = (loss_tensor * mask).sum() / mask.sum()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Évaluation sur le set de validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        f1_total = compute_f1(all_labels, all_preds)
        f1_no_autre = compute_f1(all_labels, all_preds, exclude_autre=True)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss:.4f}, "
              f"F1 (all): {f1_total:.4f}, F1 (sans AUTRE): {f1_no_autre:.4f}")

        if f1_total > best_f1:
            best_f1 = f1_total
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Nouveau meilleur modèle sauvegardé")

    generate_submission(model, df, transform=val_transform, output_csv="submission.csv")

if __name__ == "__main__":
    main()
