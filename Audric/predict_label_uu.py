import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from collections import defaultdict
from tensorflow.keras import layers, models
import math
from collections import defaultdict
from tensorflow.keras.models import load_model, Model
import xgboost as xgb


n = '4'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

class BalancedAugmentationGenerator(Sequence):
    """
    Générateur Keras qui :
      - Charge les images depuis un dossier (ex: 'raws crops') et un fichier CSV contenant
        les colonnes: label1,label2,label3,label4,project,img_name,id.
      - Calcule le label majoritaire parmi label1, label2, label3, label4.
      - Construit un dictionnaire classe -> liste d'images.
      - Pour chaque époque, génère un ensemble équilibré en oversamplant chaque classe
        pour que toutes les classes soient représentées de manière égale.
      - Applique une data augmentation (flip horizontal, rotation, variation de luminosité et de contraste)
        à toutes les images et redimensionne les images en img_size (ex: 1024x1024).
      - Retourne à chaque itération un batch (batch_size, H, W, 3) et (batch_size,) avec les labels.
    """
    
    def __init__(self, data_dir, csv_path,
                 batch_size=16,
                 img_size=(512, 512),
                 augment=True,
                 rot_range=15,
                 flip_prob=0.5):
        """
        :param data_dir: dossier contenant les images
        :param csv_path: chemin vers le CSV avec les colonnes spécifiées
        :param batch_size: taille du batch
        :param img_size: taille de redimensionnement (largeur, hauteur)
        :param augment: appliquer ou non l'augmentation
        :param rot_range: plage de rotation en degrés
        :param flip_prob: probabilité de flip horizontal
        """
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.rot_range = rot_range
        self.flip_prob = flip_prob
        
        # Dictionnaire classe -> liste d'image (chemin complet)
        self.class_to_paths = {}
        
        # Lire le CSV et construire le mapping
        self._load_csv()
        
        if len(self.class_to_paths) == 0:
            raise ValueError("Aucune image chargée. Vérifiez votre CSV et data_dir !")
        
        # Liste des classes (triée pour la reproductibilité)
        self.class_ids = sorted(self.class_to_paths.keys())
        
        # Pour équilibrer, on définit le nombre d'exemples par classe = max(count)
        self.max_count = max(len(paths) for paths in self.class_to_paths.values())
        
        # Construction de l'index global pour l'époque (liste de tuples (img_path, label))
        self.on_epoch_end()
    
    def _load_csv(self):
        """Lit le CSV et construit le dictionnaire classe -> liste d'images."""
        if not os.path.exists(self.csv_path):
            raise ValueError(f"Le CSV '{self.csv_path}' n'existe pas.")
            
        df = pd.read_csv(self.csv_path)
        # On s'assure que les colonnes nécessaires existent
        required_cols = {"label1", "label2", "label3", "label4", "img_name"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Le CSV doit contenir les colonnes {required_cols}")
            
        for _, row in df.iterrows():
            try:
                labels = [float(row["label1"]),
                          float(row["label2"]),
                          float(row["label3"]),
                          float(row["label4"])]
            except Exception as e:
                continue
            
            maj_label = self.majority_label(labels)
            img_name = row["img_name"]
            img_path = os.path.join(self.data_dir, img_name)
            if os.path.exists(img_path):
                # Ajouter l'image dans la liste de sa classe
                self.class_to_paths.setdefault(maj_label, []).append(img_path)
    
    def majority_label(self, labels):
        """
        Renvoie le label majoritaire parmi 4 labels.
        En cas d'ex-aequo, retourne le maximum.
        """
        return max(labels, key=labels.count)
    
    def on_epoch_end(self):
        """
        Pour chaque époque, on génère une liste équilibrée d'indices.
        Pour chaque classe, on tire aléatoirement (avec remplacement)
        self.max_count exemples et on combine le tout en mélangeant.
        """
        self.indexes = []
        for label, paths in self.class_to_paths.items():
            # Oversampling avec remplacement pour atteindre self.max_count
            sampled = [random.choice(paths) for _ in range(self.max_count)]
            # Pour chaque image, on stocke le tuple (img_path, label)
            self.indexes += [(p, label) for p in sampled]
        random.shuffle(self.indexes)
    
    def __len__(self):
        """Nombre de batches par époque."""
        return math.ceil(len(self.indexes) / self.batch_size)
    
    def __getitem__(self, idx):
        """
        Retourne le batch (X, y) correspondant.
        """
        batch_indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_batch = []
        y_batch = []
        for img_path, label in batch_indexes:
            img = self.load_and_transform(img_path)
            X_batch.append(img)
            y_batch.append(label)
        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.float32)
        return X_batch, y_batch
    
    def load_and_transform(self, img_path):
        """Charge l'image, la redimensionne et applique l'augmentation si activée."""
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement de l'image {img_path}: {e}")
        img = img.resize(self.img_size, Image.BICUBIC)
        if self.augment:
            img = self.random_augmentation(img)
        img_np = np.array(img, dtype=np.float32) / 255.0
        return img_np
    
    def random_augmentation(self, pil_img):
        """Applique flip horizontal, rotation, variation de luminosité et de contraste."""
        # Flip horizontal
        if random.random() < self.flip_prob:
            pil_img = ImageOps.mirror(pil_img)
        # Rotation aléatoire
        angle = random.uniform(-self.rot_range, self.rot_range)
        pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=False)
        # Variation de luminosité
        brightness_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_factor)
        # Variation de contraste
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast_factor)
        return pil_img


data_dir="../../barrage/grp3/crops/raw_crops/"
csv_path="../../barrage/grp3/crops/raw_crops/labels.csv"

generator = BalancedAugmentationGenerator(
    data_dir=data_dir,
    csv_path=csv_path,
    augment=True,
    rot_range=15,
    flip_prob=0.5,
)

resnet_full = load_model("/home/barrage/grp3/resnet101_classes_DA.h5")

# Créer un extracteur de features en coupant juste après la couche GlobalAveragePooling2D.
# Ici on suppose que la couche GAP est la troisième dernière couche.
resnet_feat_model = Model(
    inputs=resnet_full.input,
    outputs=resnet_full.layers[1].output  # Par exemple, la sortie du GAP, de dimension (None, 2048)
)

X_train_feats = []
y_train = []

# Boucler sur tout le générateur (tel que défini par steps_per_epoch)
for step_idx in range(len(generator)):
    batch_x, batch_y = generator[step_idx]
    feats = resnet_feat_model.predict(batch_x, verbose=0)  # shape: (batch_size, feature_dim)
    X_train_feats.append(feats)
    y_train.append(batch_y)

X_train_feats = np.concatenate(X_train_feats, axis=0)
y_train = np.concatenate(y_train, axis=0)
print("Taille finale X_train_feats =", X_train_feats.shape)
print("Taille finale y_train =", y_train.shape)


xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softprob',  # Utilise softmax pour prédire les probabilités pour chaque classe
    num_class=9,                # Indique le nombre de classes
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_jobs=4  # Limite l'utilisation des CPU à 4 threads
)

xgb_model.fit(X_train_feats, y_train)
print("Modèle XGBoost entraîné et sauvegardé.")

# ===========================
# 4. Extraction des features et prédiction sur le jeu de test
# ===========================
IMG_SIZE = (512, 512)

class TestFeatureGenerator(Sequence):
    def __init__(self,
                 test_dir,
                 feature_extractor,
                 batch_size=4,
                 img_size=(512, 512)):
        """
        test_dir         : dossier contenant les images à prédire
        feature_extractor: un modèle Keras qui prend en entrée (batch, h, w, 3)
                           et renvoie un tenseur de features (ex: ResNet sans la dernière Dense)
        batch_size       : taille du batch
        img_size         : taille (H, W) pour le redimensionnement des images
        """
        self.test_dir = test_dir
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Liste de tous les fichiers images
        self.files = [
            f for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        # On peut trier pour un ordre déterministe si on veut
        self.files.sort()
        
        self.n = len(self.files)  # nombre total d'images

    def __len__(self):
        # Nombre total de batches = ceil(n / batch_size)
        return math.ceil(self.n / self.batch_size)

    def __getitem__(self, index):
        """
        Retourne (X_feats_batch, ids_batch).
        - X_feats_batch : array numpy de shape (batch, dim_feat) ou (batch, H', W', C') selon la sortie du feature_extractor
        - ids_batch     : liste des noms de fichiers (IDs) correspondants
        """
        # Sélectionne un sous-ensemble de fichiers pour ce batch
        batch_files = self.files[index * self.batch_size : (index + 1) * self.batch_size]
        
        images = []
        ids_batch = []
        
        for fname in batch_files:
            path = os.path.join(self.test_dir, fname)
            # 1) Charger et convertir en RGB
            img = Image.open(path).convert("RGB")
            # 2) Redimension
            img = img.resize(self.img_size, Image.BICUBIC)
            # 3) Convertir en numpy, normaliser [0..1]
            img_np = np.array(img, dtype=np.float32) / 255.0
            # 4) Ajouter un axe batch => (1, h, w, 3)
            images.append(img_np)
            ids_batch.append(os.path.splitext(fname)[0])
        
        # Empile toutes les images du batch => shape (batch_size, h, w, 3)
        X_batch = np.stack(images, axis=0)

        # Extraire les features via le réseau
        # shape du résultat dépend de la sortie de feature_extractor
        X_feats = self.feature_extractor.predict(X_batch, verbose=0)
        
        # X_feats sera shape (batch_size, dim_features)
        return X_feats, ids_batch


TEST_DIR = "/home/barrage/grp3/datatest/"

test_generator = TestFeatureGenerator(
    test_dir=TEST_DIR,
    feature_extractor=resnet_feat_model,
    batch_size=16,
    img_size=(512, 512)
)

all_feats = []
all_ids = []

# On itère sur le générateur
for X_feats_batch, ids_batch in test_generator:
    # X_feats_batch => array shape (batch_size, feature_dim)
    all_feats.append(X_feats_batch)
    all_ids.extend(ids_batch)

# Concatène en un seul tableau
X_test_feats = np.concatenate(all_feats, axis=0)

print("Dimensions finales des features :", X_test_feats.shape)
print("Exemple d'IDs récupérés :", all_ids[:5])

# Prédiction via XGBoost
y_pred_probs = xgb_model.predict_proba(X_test_feats)
y_pred = np.argmax(y_pred_probs, axis=1)

df_preds = pd.DataFrame({
    "idx": all_ids,
    "gt": y_pred
})
df_preds.to_csv("xgb_predictions_uniquelabel_DA.csv", index=False)

print("✅ Prédictions terminées avec ResNet50 + XGBoost")
print(df_preds.head())