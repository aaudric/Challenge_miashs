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

# ---------- Configuration ressources (1 GPU, <=8 CPU) ----------
n = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
torch.set_num_threads(4)

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

# ---------- Fonction de fusion des labels (vote majoritaire) ----------
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

# ---------- Dataset pour l'entraînement/validation ----------
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

# ---------- Entraînement avec early stopping et scheduler ----------
def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=30, device='cuda', early_stopping_patience=5, use_mixup=True):
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
        scheduler.step()
        if epochs_without_improve >= early_stopping_patience:
            print("⚡ Early stopping activé")
            break
    model.load_state_dict(best_model_wts)
    return model

# ---------- Phase d'entraînement ----------
def main_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Lecture du CSV d'entraînement et fusion des labels
    df = pd.read_csv(args.csv_labels)
    df['label'] = df.apply(fusion_labels, axis=1)
    print("Distribution initiale des classes :")
    print(df['label'].value_counts())

    # Rééquilibrer le dataset pour avoir 100 images par classe
    df_balanced = balance_dataset(df, target_count=100, label_col='label', random_state=42)
    print("Distribution après équilibrage :")
    print(df_balanced['label'].value_counts())

    # Transformations d'augmentation (spatiales)
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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

    # Utilisation de la validation croisée (StratifiedKFold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_balanced, df_balanced['label'])):
        print(f"\n========== Fold {fold+1} ==========")
        train_data = TrainImageDataset(df_balanced.iloc[train_idx], args.img_dir, transform=transforms_train)
        val_data = TrainImageDataset(df_balanced.iloc[val_idx], args.img_dir, transform=transforms_val)
        dataloaders = {
            'train': DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        }
        # Création du modèle
        model = ResNet50WithSE(num_classes=9, hidden_size=512, dropout=0.5, reduction=16)
        model = model.to(device)

        # Calcul des poids pour la Focal Loss (basé sur df_balanced)
        class_counts = df_balanced['label'].value_counts().sort_index().values
        weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        weights = weights / weights.sum()
        weights = weights.to(device)
        criterion = FocalLoss(alpha=weights, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        model = train_model(model, criterion, optimizer, scheduler, dataloaders,
                            num_epochs=args.epochs, device=device,
                            early_stopping_patience=args.early_stopping, use_mixup=args.use_mixup)
        fold_path = f"best_model_fold{fold+1}.pth"
        torch.save(model.state_dict(), fold_path)
        print(f"📦 Modèle sauvegardé : {fold_path}")

# ---------- Phase d'inférence (Test) avec TTA ----------
def main_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TTA avec TenCrop
    tta_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),  # 10 recadrages
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ])
    test_dataset = TestFolderDataset(args.test_img_dir, transform=tta_transform, extension=args.test_ext)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Charger les modèles sauvegardés (issus des 5 folds)
    fold_models = []
    for fold in range(1, 6):
        model = ResNet50WithSE(num_classes=9, hidden_size=512, dropout=0.5, reduction=16)
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
            bs, ncrops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w)
            inputs = inputs.to(device)
            outputs_sum = torch.zeros(bs, 9).to(device)
            for model in fold_models:
                outputs = model(inputs)
                outputs = outputs.view(bs, ncrops, -1).mean(1)
                outputs_sum += torch.softmax(outputs, dim=1)
            outputs_avg = outputs_sum / len(fold_models)
            preds = outputs_avg.argmax(dim=1)
            all_predictions.extend(preds.cpu().numpy())
            image_names.extend(base_names)
    df_result = pd.DataFrame({"img_name": image_names, "predictions": all_predictions})
    output_csv = "predictions_test.csv"
    df_result.to_csv(output_csv, index=False)
    print(f"📦 Prédictions sauvegardées dans {output_csv}")

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
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--early_stopping', type=int, default=5, help="Patience pour l'early stopping")
    parser.add_argument('--use_mixup', type=bool, default=False, help="Activer ou non le MixUp pendant l'entraînement")
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
        if not args.csv_labels or not args.img_dir or not args.test_img_dir:
            raise ValueError("Pour le mode 'all', --csv_labels, --img_dir et --test_img_dir sont requis.")
        main_train(args)
        main_test(args)
