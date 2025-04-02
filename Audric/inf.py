import os
import argparse
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Configuration multi-threading et device
n = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3})

IMG_SIZE = 224
NUM_CLASSES = 9  # On conserve 9 sorties, mais la classe 3 sera ignorée
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- TRANSFORM -----
# Pour l'inférence, on se contente d'un redimensionnement et de la normalisation
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----- DATASET -----
class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = sorted([f for f in os.listdir(img_dir)
                                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Retourne également l'identifiant (nom de fichier sans extension)
        return image, os.path.splitext(img_name)[0]

# ----- MODEL -----
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

# ----- PREDICTION -----
def predict(model, dataloader, output_csv="submission_finale.csv"):
    results = []
    # Définition des seuils par classe : 0.7 pour la classe 0, 0.5 pour les autres
    thresholds = np.array([0.7] + [0.5] * (NUM_CLASSES - 1))
    
    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()  # Probabilités de forme (batch, 9)
            
            # Appliquer les seuils
            preds = (probs > thresholds).astype(int)
            # Forcer la classe 3 à être ignorée
            preds[:, 3] = 0

            # Pour chaque image, si aucune classe n'est retenue, sélectionner la meilleure alternative
            for i in range(len(preds)):
                if not preds[i].any():
                    # Exclure la classe 3 en mettant sa probabilité à -inf
                    temp = probs[i].copy()
                    temp[3] = -np.inf
                    idx_sel = np.argmax(temp)
                    preds[i][idx_sel] = 1

            # Construction des résultats
            for idx, p in zip(ids, preds):
                # On renvoie la liste des indices des classes prédits
                pred_labels = np.where(p == 1)[0]
                results.append({'idx': idx, 'gt': '[' + ', '.join(str(i) for i in pred_labels) + ']'})
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Fichier de soumission généré : {output_csv}")

# ----- MAIN -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help='Chemin vers le dossier contenant les images')
    parser.add_argument('--weights', type=str, required=True, help='Chemin vers le fichier best_model.pth')
    parser.add_argument('--output', type=str, default='submission_finale.csv', help='Nom du fichier de soumission')
    args = parser.parse_args()
    
    dataset = TestImageDataset(args.img_dir, transform=inference_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = load_model(args.weights)
    predict(model, dataloader, args.output)

if __name__ == "__main__":
    main()
