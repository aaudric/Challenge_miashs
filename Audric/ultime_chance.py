import os
import argparse
import copy
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image, ImageEnhance, ImageOps
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

# ---------- Configuration ressources (1 GPU, <=8 CPU) ----------
n = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3})
torch.set_num_threads(4)

# ---------- Transformation personnalisée : Enlarge, Flip, Crop, AutoContrast ----------
class EnlargeFlipCropAutoContrast(object):
    def __init__(self, target_size=(224, 224), scale_factor=1.5, flip_prob=0.5):
        """
        - Agrandit l'image en ajoutant un padding en mode 'reflect' pour atteindre une taille égale à scale_factor * target_size.
        - Effectue un flip horizontal aléatoire avec probabilité flip_prob.
        - Recadre aléatoirement l'image agrandie pour obtenir la taille target_size (ici 224×224).
        - Applique un autocontraste pour uniformiser le contraste de toutes les images.
        """
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.flip_prob = flip_prob
        # Calculer le padding (pour chaque côté) : ex. pour 224 et scale_factor 1.5, pad = 224*(1.5-1)/2 = 56
        self.pad = int(target_size[0] * (scale_factor - 1) / 2)

    def __call__(self, img):
        # Convertir l'image en tableau numpy
        np_img = np.array(img)
        # Ajouter un padding en mode 'reflect' pour récupérer le contexte (pas de zones noires)
        pad_width = ((self.pad, self.pad), (self.pad, self.pad), (0, 0))
        np_img_padded = np.pad(np_img, pad_width, mode='reflect')
        img_padded = Image.fromarray(np_img_padded)
        # Flip horizontal aléatoire
        if np.random.rand() < self.flip_prob:
            img_padded = img_padded.transpose(Image.FLIP_LEFT_RIGHT)
        # Recadrage aléatoire pour obtenir la taille cible
        padded_w, padded_h = img_padded.size
        target_w, target_h = self.target_size
        left = np.random.randint(0, padded_w - target_w + 1)
        top = np.random.randint(0, padded_h - target_h + 1)
        img_cropped = img_padded.crop((left, top, left + target_w, top + target_h))
        # Appliquer autocontraste pour uniformiser le contraste
        img_final = ImageOps.autocontrast(img_cropped, cutoff=0)
        return img_final

# ---------- Fonction de rééquilibrage du dataset ----------
def balance_dataset(df, target_count=100, label_col='label', random_state=42):
    """
    Retourne un DataFrame équilibré où chaque classe contient target_count images.
    Si une classe contient moins de target_count images, échantillonnage avec replacement.
    """
    balanced_groups = []
    for label, group in df.groupby(label_col):
        if len(group) < target_count:
            sample = group.sample(n=target_count, replace=True, random_state=random_state)
        else:
            sample = group.sample(n=target_count, replace=False, random_state=random_state)
        balanced_groups.append(sample)
    balanced_df = pd.concat(balanced_groups).reset_index(drop=True)
    return balanced_df

# ---------- Fusion des labels (vote majoritaire) ----------
def fusion_labels(row):
    votes = [row[f'label{i}'] for i in range(1, 5)]
    return int(Counter(votes).most_common(1)[0][0])

# ---------- Focal Loss ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
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

# ---------- MixUp ----------
def mixup_data(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---------- SE Block (Squeeze-and-Excitation) ----------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ---------- Architecture : ResNet50 avec SE ----------
class ResNet50WithSE(nn.Module):
    def __init__(self, num_classes=9, hidden_size=512, dropout=0.5, reduction=16):
        super(ResNet50WithSE, self).__init__()
        base = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4
        )
        self.se = SEBlock(2048, reduction=reduction)
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ---------- Dataset pour l'entraînement ----------
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

# ---------- Dataset pour le test (à partir du dossier) ----------
class TestFolderDataset(Dataset):
    def __init__(self, img_dir, transform=None, extension=".jpg"):
        self.img_dir = img_dir
        self.transform = transform
        self.extension = extension
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(extension)]
        self.img_files.sort()  # pour la reproductibilité

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        base_name = os.path.splitext(img_name)[0]
        return img, base_name

# ---------- Entraînement sur tout le jeu (sans validation) ----------
def train_model_all(model, criterion, optimizer, scheduler, dataloader, num_epochs=30, device='cuda', use_mixup=False):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_mixup:
                inputs_mix, y_a, y_b, lam = mixup_data(inputs, labels, alpha=0.4)
                outputs = model(inputs_mix)
                loss = mixup_loss(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - F1: {epoch_f1:.4f}")
        scheduler.step()
    return model

# ---------- Phase d'entraînement sur tout le train en stratified K-fold ----------
def train_kfold(model_class, df, img_dir, args, device):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    # Construction du tableau multilabel pour la stratification
    multilabels = np.zeros((len(df), 9))
    for i, row in df.iterrows():
        # Ici, la colonne 'labels' est supposée contenir une liste d'indices
        for label in row['labels']:
            multilabels[i, label] = 1

    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    fold = 1
    for train_idx, val_idx in mskf.split(df['img_name'], multilabels):
        print(f"\n========== Fold {fold} ==========")
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        # Rééquilibrer le DataFrame d'entraînement pour avoir 100 images par classe
        df_train_bal = balance_dataset(df_train, target_count=100, label_col='labels', random_state=42)
        print("Distribution après équilibrage (train fold) :")
        print(df_train_bal['labels'].apply(len).value_counts())
        
        transforms_train = transforms.Compose([
            EnlargeFlipCropAutoContrast(target_size=(224, 224), scale_factor=1.5, flip_prob=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        transforms_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        
        train_dataset = MultiLabelDataset(df_train_bal, img_dir, transform=transforms_train)
        val_dataset = MultiLabelDataset(df_val, img_dir, transform=transforms_val)
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        }
        
        model = model_class(num_classes=9, hidden_size=512, dropout=0.5, reduction=16)
        model = model.to(device)
        
        # Calcul des poids pour la Focal Loss sur df_train_bal
        # Ici, on compte le nombre d'images par classe à partir des labels fusionnés
        # Pour les problèmes multi-label, cela peut être adapté selon vos besoins.
        class_counts = np.zeros(9)
        for labels in df_train_bal['labels']:
            for l in labels:
                class_counts[l] += 1
        class_counts = torch.tensor(class_counts, dtype=torch.float)
        weights = 1. / (class_counts + 1e-6)
        weights = weights / weights.sum()
        weights = weights.to(device)
        
        criterion = FocalLoss(alpha=weights, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_f1 = 0.0
        epochs_without_improve = 0
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            train_preds, train_labels = [], []
            for inputs, labels in dataloaders['train']:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            train_loss = running_loss / len(dataloaders['train'].dataset)
            
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            val_loss /= len(dataloaders['val'].dataset)
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            print(f"Fold {fold} - Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_without_improve = 0
                print("✅ Nouveau meilleur modèle pour ce fold")
            else:
                epochs_without_improve += 1
            scheduler.step()
            if epochs_without_improve >= args.early_stopping:
                print("⚡ Early stopping sur ce fold")
                break
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
        print(f"📦 Modèle du fold {fold} sauvegardé sous 'best_model_fold{fold}.pth'")
        fold += 1

# ---------- Transformation personnalisée mise à jour ----------
# Renommée ici pour clarté
class EnlargeFlipCropAutoContrast(object):
    def __init__(self, target_size=(224, 224), scale_factor=1.5, flip_prob=0.5):
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.flip_prob = flip_prob
        self.pad = int(target_size[0] * (scale_factor - 1) / 2)

    def __call__(self, img):
        # Convertir en numpy array
        np_img = np.array(img)
        pad_width = ((self.pad, self.pad), (self.pad, self.pad), (0, 0))
        np_img_padded = np.pad(np_img, pad_width, mode='reflect')
        img_padded = Image.fromarray(np_img_padded)
        if np.random.rand() < self.flip_prob:
            img_padded = img_padded.transpose(Image.FLIP_LEFT_RIGHT)
        padded_w, padded_h = img_padded.size
        target_w, target_h = self.target_size
        left = np.random.randint(0, padded_w - target_w + 1)
        top = np.random.randint(0, padded_h - target_h + 1)
        img_cropped = img_padded.crop((left, top, left + target_w, top + target_h))
        # Appliquer autocontraste pour uniformiser le contraste
        img_final = ImageOps.autocontrast(img_cropped, cutoff=0)
        return img_final

# ---------- Dataset pour l'entraînement multi-label ----------
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
            label = torch.zeros(NUM_CLASSES)
        else:
            label_indices = row['labels']
            label = torch.zeros(NUM_CLASSES)
            label[label_indices] = 1.0
        return image, label

# ---------- Phase d'inférence (Test) avec TTA ----------
def main_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),  # 10 recadrages
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ])
    test_dataset = TestFolderDataset(args.test_img_dir, transform=tta_transform, extension=args.test_ext)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    fold_models = []
    for fold in range(1, 6):
        model = ResNet50WithSE(num_classes=9, hidden_size=512, dropout=0.5, reduction=16)
        model_path = f"best_model_fold{fold}.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            fold_models.append(model)
            print(f"✅ Modèle du fold {fold} chargé")
        else:
            print(f"⚠️ Le modèle {model_path} n'existe pas")
    all_predictions = []
    image_names = []
    with torch.no_grad():
        for inputs, base_names in test_loader:
            bs, ncrops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w).to(device)
            outputs_sum = torch.zeros(bs, 9).to(device)
            for model in fold_models:
                outputs = model(inputs)
                outputs = outputs.view(bs, ncrops, -1).mean(1)
                outputs_sum += torch.softmax(outputs, dim=1)
            outputs_avg = outputs_sum / len(fold_models)
            preds = outputs_avg.argmax(dim=1)
            all_predictions.extend(preds.cpu().numpy())
            image_names.extend(base_names)
    df_result = pd.DataFrame({"idx": image_names, "gt": all_predictions})
    output_csv = "predictions_test.csv"
    df_result.to_csv(output_csv, index=False)
    print("📦 Prédictions sauvegardées dans 'predictions_test.csv'.")

# ---------- Programme principal ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'all'], required=True,
                        help="Mode d'exécution : 'train', 'test' ou 'all'.")
    # Arguments pour l'entraînement
    parser.add_argument('--csv_labels', type=str, help="Chemin vers le CSV d'entraînement")
    parser.add_argument('--img_dir', type=str, help="Dossier contenant les images d'entraînement")
    # Arguments pour le test
    parser.add_argument('--test_img_dir', type=str, help="Dossier contenant les images de test")
    parser.add_argument('--test_ext', type=str, default=".jpg", help="Extension des images de test (ex: .jpg)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--early_stopping', type=int, default=5, help="Patience pour l'early stopping (fold)")
    parser.add_argument('--use_mixup', type=bool, default=False, help="Activer ou non le MixUp pendant l'entraînement")
    args = parser.parse_args()

    if args.mode in ['train', 'all']:
        df = pd.read_csv(args.csv_labels)
        # Fusionner les labels des 4 experts pour obtenir une liste multi-label
        def fusion_labels(row):
            labels = [int(row[f'label{i}']) for i in range(1, 5)]
            return list(set(labels))
        df['labels'] = df.apply(fusion_labels, axis=1)
        print("Distribution initiale des classes (selon le nombre de labels par image) :")
        print(df['labels'].apply(len).value_counts())
        # On équilibre le DataFrame d'entraînement sur la base des labels multi-label (selon votre stratégie)
        df_balanced = balance_dataset(df, target_count=100, label_col='labels', random_state=42)
        print("Distribution après équilibrage (nombre d'images par label par image) :")
        print(df_balanced['labels'].apply(len).value_counts())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_kfold(ResNet50WithSE, df_balanced, args.img_dir, args, device)
    if args.mode in ['test', 'all']:
        main_test(args)
