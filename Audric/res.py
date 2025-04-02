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
from tensorflow.keras.applications import ResNet50


n = '4'

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)


from collections import defaultdict, Counter

import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
from tensorflow.keras.utils import Sequence

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



background_dir="../../barrage/grp3/crops/background_patches/"
data_dir="../../barrage/grp3/crops/raw_crops/"
csv_path="../../barrage/grp3/crops/raw_crops/labels.csv"

generator = BalancedAugmentationGenerator(
    data_dir=data_dir,
    csv_path=csv_path,
    augment=True,
    rot_range=15,
    flip_prob=0.5,  # on choisit qu'une epoch fasse 100 steps
)

def build_resnet50_classifier(num_classes=9):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(None, None, 3)
    )
    # Fine-tuning : on peut mettre base_model.trainable = False 
    # si on veut entraîner seulement la partie Dense au début
    base_model.trainable = True  

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512),
        layers.PReLU(),
        layers.Dense(512),
        layers.PReLU(),
        layers.Dense(num_classes, activation='softmax')  # 9 classes
    ])

    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

    return model

    # Construction du modèle
model = build_resnet50_classifier(num_classes=9)

    # Entraînement
    # nb : si vous avez aussi un générateur de validation, vous pouvez le passer en `validation_data=val_generator`
model.fit(
        generator,
        epochs=10
    )

model.save("../../barrage/grp3/resnet101_classes_DA.h5")
print("Modèle sauvegardé sous 'resnet101_9classes.h5'")