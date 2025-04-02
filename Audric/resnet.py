import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageOps, ImageEnhance
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, PReLU



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
n = '4'

os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 

# =============================================================================
# Générateur personnalisé avec data augmentation (pour entraînement et validation)
# =============================================================================
class BalancedAugmentationGenerator(Sequence):
    def __init__(self, df, img_dir, batch_size=8, augment=False, img_size=(1024, 1024), shuffle=True):
        """
        df         : DataFrame contenant au moins les colonnes 'img_name' et 'label'
        img_dir    : Dossier contenant les images
        batch_size : Taille du batch
        augment    : Booléen, appliquer ou non l'augmentation (seulement pour training)
        img_size   : Taille de redimensionnement (H, W)
        shuffle    : Mélanger les données à la fin de chaque époque
        """
        self.df = df.copy()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.augment = augment
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        # Sélection des indices pour ce batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]
        
        X = []
        y = []
        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.img_dir, row['img_name'])
            if not os.path.exists(img_path):
                continue
            # Charger l'image et la redimensionner
            img = load_img(img_path, target_size=self.img_size)
            img = img_to_array(img)
            img = img / 255.0  # Normalisation
            if self.augment:
                img = self.random_augmentation(img)
            X.append(img)
            y.append(int(row['label']))
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def random_augmentation(self, img):
        """Applique flip horizontal, rotation, variation de luminosité et de contraste."""
        # Convertir le tableau numpy en image PIL
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        # Flip horizontal avec probabilité 0.5
        if random.random() < 0.5:
            pil_img = ImageOps.mirror(pil_img)
        # Rotation aléatoire dans [-15, 15] degrés
        angle = random.uniform(-15, 15)
        pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=False)
        # Variation de luminosité
        brightness_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_factor)
        # Variation de contraste
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast_factor)
        # Retourner l'image normalisée en numpy array
        img = np.array(pil_img, dtype=np.float32) / 255.0
        return img

# =============================================================================
# Chargement et préparation des données
# =============================================================================
csv_path = "data/grp3/crops/raw_crops/labels.csv"       # chemin vers le CSV
img_dir = "data/grp3/crops/raw_crops/"         # dossier contenant les images

# Charger le CSV
df = pd.read_csv(csv_path)

# Filtrer pour ne conserver que les lignes où tous les labels sont identiques
df = df[(df['label1'] == df['label2']) & (df['label2'] == df['label3']) & (df['label3'] == df['label4'])]
# Créer une colonne 'label' qui est le label majoritaire (ici identique pour toutes les colonnes)
df['label'] = df['label1'].astype(int)

print("Nombre total d'occurrences par label :")
print(df['label'].value_counts())

# =============================================================================
# Séparation stratifiée en train et validation
# =============================================================================
train_df = (df, stratify=df['label'])

# =============================================================================
# Création des générateurs
# =============================================================================
train_gen = BalancedAugmentationGenerator(train_df, img_dir, batch_size=16, augment=True, img_size=(512,512), shuffle=True)

# =============================================================================
# Calcul des class weights pour compenser l'inéquilibre
# =============================================================================
classes = np.unique(train_df['label'])
class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=train_df['label'])
class_weights_dict = {int(c): w for c, w in zip(classes, class_weights)}
print("Class weights :", class_weights_dict)

# =============================================================================
# Construction du modèle ResNet50 avec régularisation (dropout)
# =============================================================================
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(512,512,3))
# On peut choisir de geler la base pour débuter
base_model.trainable = True

inputs = Input(shape=(512,512,3))
x = base_model(inputs, training=True)
x = Dropout(0.5)(x)  # Ajout de dropout pour limiter le surapprentissage
outputs = Dense(9, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# =============================================================================
# Entraînement du modèle
# =============================================================================
# Ajouter à tes imports précédents
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Création d'un Callback personnalisé pour calculer le F1-score à chaque époque
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_gen, val_gen):
        super().__init__()
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.train_f1 = []
        self.val_f1 = []

    def on_epoch_end(self, epoch, logs=None):
        # Prédictions sur train
        train_preds = []
        train_labels = []
        for X_batch, y_batch in self.train_gen:
            preds = self.model.predict(X_batch, verbose=0)
            train_preds.extend(np.argmax(preds, axis=1))
            train_labels.extend(y_batch)
        
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        self.train_f1.append(train_f1)
        
        # Prédictions sur validation
        val_preds = []
        val_labels = []
        for X_batch, y_batch in self.val_gen:
            preds = self.model.predict(X_batch, verbose=0)
            val_preds.extend(np.argmax(preds, axis=1))
            val_labels.extend(y_batch)
        
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        self.val_f1.append(val_f1)
        
        print(f"Epoch {epoch+1}: Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

# Initialiser le callback F1-score
f1_callback = F1ScoreCallback(train_gen, val_gen)

# Entraînement du modèle avec le callback F1-score
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    class_weight=class_weights_dict,
    callbacks=[
        f1_callback,
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("resnet50_finetuned.h5", monitor='val_accuracy', save_best_only=True)
    ]
)

# Sauvegarde finale du modèle
model.save("resnet50_final.h5")

# =============================================================================
# Plot des Loss et F1-score
# =============================================================================
epochs_range = range(len(history.history['loss']))

plt.figure(figsize=(14, 5))

# Plot des Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'], label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss en fonction des époques')
plt.save("loss.png")

# Plot des F1-scores
plt.subplot(1, 2, 2)
plt.plot(epochs_range, f1_callback.train_f1, label='Train F1-score')
plt.plot(epochs_range, f1_callback.val_f1, label='Val F1-score')
plt.legend()
plt.title('F1-score en fonction des époques')
plt.save('F1.png')

plt.show()

