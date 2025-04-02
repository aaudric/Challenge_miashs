import os
import argparse
import copy
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

# ---------------- Réglages généraux ----------------
n = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3})
torch.set_num_threads(4)

# ---------------- Focal Loss ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # S'assurer que alpha est du même type que inputs
        weight = self.alpha.to(inputs.dtype) if self.alpha is not None else None
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ---------------- Dataset pour l'entraînement/validation ----------------
class TrainImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['img_name'])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = row['label']
        return img, label

# ---------------- Dataset pour le test à partir du dossier ----------------
class TestFolderDataset(Dataset):
    def __init__(self, img_dir, transform=None, extension=".jpg"):
        self.img_dir = img_dir
        self.transform = transform
        self.extension = extension
        # Récupère tous les fichiers se terminant par l'extension spécifiée
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(extension)]
        self.img_files.sort()  # Tri pour garantir la reproductibilité

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        base_name = os.path.splitext(img_name)[0]  # nom sans extension
        return img, base_name

# ---------------- Fusion des labels par vote majoritaire ----------------
def fusion_labels(row):
    votes = [row[f'label{i}'] for i in range(1, 5)]
    # Convertir le résultat en entier
    return int(Counter(votes).most_common(1)[0][0])


# ---------------- MixUp ----------------
def mixup_data(x, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))  # Convertir en float Python
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---------------- Entraînement avec early stopping ----------------
def train_model(model, criterion, optimizer, dataloaders, num_epochs=25, device='cuda', early_stopping_patience=5, use_mixup=True):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    epochs_without_improve = 0

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train' and use_mixup:
                        inputs_mix, y_a, y_b, lam = mixup_data(inputs, labels, alpha=0.4)
                        outputs = model(inputs_mix)
                        loss = mixup_loss(criterion, outputs, y_a, y_b, lam)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')
            print(f"{phase} Loss: {epoch_loss:.4f} | F1: {epoch_f1:.4f}")
            if phase == 'val':
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_without_improve = 0
                    print("✅ Nouveau meilleur modèle (F1)")
                else:
                    epochs_without_improve += 1
        if epochs_without_improve >= early_stopping_patience:
            print("⚡ Early stopping activé")
            break
    model.load_state_dict(best_model_wts)
    return model


# ---------------- Phase d'entraînement ----------------
def main_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Lecture du CSV d'entraînement et fusion des labels
    df = pd.read_csv(args.csv_labels)
    df['label'] = df.apply(fusion_labels, axis=1)

    # Transformations (aucune modification du contraste)
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n========== Fold {fold+1} ==========")
        train_data = TrainImageDataset(df.iloc[train_idx], args.img_dir, transform=transforms_train)
        val_data = TrainImageDataset(df.iloc[val_idx], args.img_dir, transform=transforms_val)
        dataloaders = {
            'train': DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        }
        # Modèle pré-entraîné et ajustement de la couche finale pour 9 classes
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features

        # Définir la nouvelle tête de classification avec 2 couches denses et dropout
        hidden_size = 512  # Vous pouvez ajuster cette taille selon vos besoins
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 9)
        )
        model = model.to(device)

        # Calcul des poids pour la Focal Loss
        class_counts = df['label'].value_counts().sort_index().values
        weights = 1. / torch.tensor(class_counts, dtype=torch.float)  # ici torch.float signifie float32
        weights = weights / weights.sum()
        weights = weights.to(device)

        criterion = FocalLoss(alpha=weights, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model = train_model(model, criterion, optimizer, dataloaders,
                            num_epochs=args.epochs, device=device,
                            early_stopping_patience=args.early_stopping, use_mixup= False )
        fold_path = f"best_model_fold{fold+1}.pth"
        torch.save(model.state_dict(), fold_path)
        print(f"📦 Modèle sauvegardé : {fold_path}")

# ---------------- Phase d'inférence (Test) avec ensemble ----------------
def main_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transformation de test (sans augmentation du contraste)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    # Création du dataset à partir du dossier de test
    test_dataset = TestFolderDataset(args.test_img_dir, transform=test_transform, extension=args.test_ext)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Charger les modèles sauvegardés issus des 5 folds
    fold_models = []
    for fold in range(1, 6):
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features

        # Définir la nouvelle tête de classification avec 2 couches denses et dropout
        hidden_size = 512  # Vous pouvez ajuster cette taille selon vos besoins
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 9)
        )
        model_path = f"best_model_fold{fold}.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            fold_models.append(model)
            print(f"✅ Modèle du fold {fold} chargé")
        else:
            print(f"⚠️ Le modèle {model_path} n'existe pas")
    all_predictions = []
    image_names = []
    with torch.no_grad():
        for inputs, base_names in test_loader:
            inputs = inputs.to(device)
            outputs_sum = torch.zeros(inputs.size(0), 9).to(device)
            for model in fold_models:
                outputs = model(inputs)
                outputs_sum += torch.softmax(outputs, dim=1)
            outputs_avg = outputs_sum / len(fold_models)
            preds = outputs_avg.argmax(dim=1)
            all_predictions.extend(preds.cpu().numpy())
            image_names.extend(base_names)
    # Création d'un DataFrame final avec les prédictions
    df_result = pd.DataFrame({"idx": image_names, "gt": all_predictions})
    output_csv = "predictions_test_ensemble2.csv"
    df_result.to_csv(output_csv, index=False)
    print(f"📦 Prédictions sauvegardées dans {output_csv}")

# ---------------- Programme principal ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'all'], required=True,
                        help="Mode d'exécution : 'train' pour entraîner, 'test' pour prédire, 'all' pour lancer les deux.")
    # Arguments pour l'entraînement
    parser.add_argument('--csv_labels', type=str, help="Chemin vers le CSV d'entraînement")
    parser.add_argument('--img_dir', type=str, help="Dossier contenant les images d'entraînement")
    # Arguments pour le test
    parser.add_argument('--test_img_dir', type=str, help="Dossier contenant les images de test")
    parser.add_argument('--test_ext', type=str, default=".jpg", help="Extension des images de test (ex: .jpg)")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--early_stopping', type=int, default=5, help="Patience pour l'early stopping")
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.csv_labels or not args.img_dir:
            raise ValueError("Pour le mode 'train', --csv_labels et --img_dir sont requis.")
        main_train(args)
    elif args.mode == 'test':
        if not args.test_img_dir:
            raise ValueError("Pour le mode 'test', --test_img_dir est requis.")
        main_test(args)
    elif args.mode == 'all':
        # Pour le mode all, on exige que les chemins pour l'entraînement et le test soient fournis
        if not args.csv_labels or not args.img_dir or not args.test_img_dir:
            raise ValueError("Pour le mode 'all', --csv_labels, --img_dir et --test_img_dir sont requis.")
        main_train(args)
        main_test(args)

