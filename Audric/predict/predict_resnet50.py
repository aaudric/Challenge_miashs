import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import xgboost as xgb
from tensorflow.keras.utils import Sequence
import random


# ===========================
# Configuration CPU/GPU
# ===========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Utiliser uniquement le GPU 0
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.sched_setaffinity(0, {0, 1, 2, 3})

TEST_DIR = "../../barrage/grp3/datatest/"

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

class PerImage4LabelsBatchGeneratorFromCSV(Sequence):
    def __init__(self, background_dir, data_dir, csv_path, batch_size=4, shuffle=True, max_background=300):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_background = max_background
        self.samples = []

        self._load_background(background_dir)
        self._load_from_csv(data_dir, csv_path)

        if self.shuffle:
            np.random.shuffle(self.samples)

    def _load_background(self, folder):
        all_imgs = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        selected_imgs = random.sample(all_imgs, min(self.max_background, len(all_imgs)))

        for img_name in selected_imgs:
            img_path = os.path.join(folder, img_name)
            txt_path = img_path.replace(".jpg", ".txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path) as f:
                line = f.readline().strip().split()
                label_str = line[0]
                labels = [int(l) for l in label_str.split("_")]
                if len(labels) == 4:
                    self.samples.append((img_path, labels))

    def _load_from_csv(self, data_dir, csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            labels = [int(row["label1"]), int(row["label2"]), int(row["label3"]), int(row["label4"])]
            img_name = row["img_name"]
            img_path = os.path.join(data_dir, img_name)
            if os.path.exists(img_path):
                self.samples.append((img_path, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, labels = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        img_np = np.array(img).astype("float32") / 255.0

        batch_images = np.stack([img_np] * 4, axis=0)
        batch_labels = np.array(labels, dtype=np.int32)

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

train_generator = PerImage4LabelsBatchGeneratorFromCSV(
    background_dir="../../barrage/grp3/crops/background_patches/",
    data_dir="../../barrage/grp3/crops/raw_crops/",
    csv_path="../../barrage/grp3/crops/raw_crops/labels.csv"
)


# ===========================
# 1. Charger le modèle ResNet50 entraîné (.h5)
# ===========================
# Ce modèle doit être entraîné pour 9 classes (avec Dense(9, activation='softmax'))
resnet_full = load_model("resnet50_9classes.h5")

# Créer un extracteur de features en coupant juste après la couche GlobalAveragePooling2D.
# Ici on suppose que la couche GAP est la troisième dernière couche.
resnet_feat_model = Model(
    inputs=resnet_full.input,
    outputs=resnet_full.layers[-3].output  # Par exemple, la sortie du GAP, de dimension (None, 2048)
)

X_train_feats = []
y_train = []

# Boucle sur le générateur pour extraire les features de chaque batch
for batch_x, batch_y in train_generator:
    feats = resnet_feat_model.predict(batch_x, verbose=0)  # shape: (batch, 2048)
    X_train_feats.append(feats)
    y_train.append(batch_y)
    # On sort de la boucle une fois que toutes les données ont été traitées
    # (ou utiliser un compteur en fonction du nombre d'itérations souhaitées)
    if len(X_train_feats) * batch_x.shape[0] >= len(train_generator) * batch_x.shape[0]:
        break

X_train_feats = np.concatenate(X_train_feats, axis=0)
y_train = np.concatenate(y_train, axis=0)



# ===========================
# 3. Entraîner XGBoost avec objective 'multi:softprob'
# ===========================
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
IMG_SIZE = (224, 224)
features_test = []
ids = []

for img_name in os.listdir(TEST_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(TEST_DIR, img_name)
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_np = np.array(img).astype("float32") / 255.0
    img_np = np.expand_dims(img_np, axis=0)  # (1, 224, 224, 3)

    feat = resnet_feat_model.predict(img_np, verbose=0)
    features_test.append(feat[0])
    ids.append(os.path.splitext(img_name)[0])

X_test_feats = np.array(features_test)
y_pred_probs = xgb_model.predict_proba(X_test_feats)
y_pred = np.argmax(y_pred_probs, axis=1)

# ===========================
# 5. Sauvegarder les prédictions dans un DataFrame CSV
# ===========================
df_preds = pd.DataFrame({
    "idx": ids,
    "gt": y_pred
})
df_preds.to_csv("xgb_predictions.csv", index=False)

print("✅ Prédictions terminées avec ResNet50 + XGBoost")
print(df_preds.head())