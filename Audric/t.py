import os
import argparse
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Configuration multi-threading
n = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3})

# --- CONFIG ---
IMG_SIZE = 224
NUM_CLASSES = 9
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FONCTION POUR AJOUTER DU CONTEXTE ---
def add_context(img, factor=1.4):
    """
    Agrandit l'image par un facteur donné en ajoutant des marges 
    avec remplissage par réflexion afin d'ajouter du contexte.
    """
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    pad_w, pad_h = new_w - w, new_h - h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return transforms.functional.pad(img, (left, top, right, bottom), padding_mode='reflect')

# --- TRANSFORM ---
val_transform = transforms.Compose([
    transforms.Lambda(lambda img: add_context(img, factor=1.4)),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- DATASET ---
class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.splitext(img_name)[0]

# --- MODEL ---
def load_model(weights_path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --- INFERENCE AVEC SEUILS CALIBRÉS ET EXCLUSION DE LA CLASSE 3 ---
def predict(model, dataloader, output_csv="submission_tt.csv"):
    results = []
    # Définition des seuils pour chaque classe (classe 0: 0.7, les autres: 0.5)
    thresholds = np.array([0.7] + [0.5] * (NUM_CLASSES - 1))
    
    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()  # Probabilités issues du modèle

            # Calcul des scores calibrés = probabilité - seuil
            calibrated_scores = probs - thresholds

            # On désactive les classes ne dépassant pas leur seuil (score négatif)
            calibrated_scores[calibrated_scores < 0] = -np.inf

            # Sélection de la classe avec le meilleur surplus
            preds = np.argmax(calibrated_scores, axis=1)

            # Si la classe prédite est 3 (interdite), on l'exclut en recalculant sans cette classe
            for i in range(len(calibrated_scores)):
                if preds[i] == 3:
                    temp_scores = calibrated_scores[i].copy()
                    temp_scores[3] = -np.inf  # Exclure la classe 3
                    new_pred = np.argmax(temp_scores)
                    preds[i] = new_pred
                # Si aucune classe ne dépasse le seuil (toutes à -inf), on choisit la classe avec la plus grande probabilité brute
                if np.all(calibrated_scores[i] == -np.inf):
                    preds[i] = np.argmax(probs[i])
                    
            for idx, label in zip(ids, preds):
                results.append({'idx': idx, 'gt': label})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Fichier de soumission généré : {output_csv}")

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help='/chemin/vers/le/dossier/images')
    parser.add_argument('--weights', type=str, required=True, help='Chemin vers le fichier best_model.pth')
    parser.add_argument('--output', type=str, default='submission_tt.csv', help='Nom du fichier de soumission')
    args = parser.parse_args()

    dataset = TestImageDataset(args.img_dir, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(args.weights)
    predict(model, dataloader, args.output)

if __name__ == "__main__":
    main()
