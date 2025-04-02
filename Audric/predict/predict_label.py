import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import xgboost as xgb
from tensorflow.keras.models import load_model, Model
import math




n = '4'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

TEST_DIR = "../../barrage/grp3/datatest/"

class PerImageBatchGeneratorFromCSV(Sequence):
    """
    Générateur Keras qui :
    - Pioche des images d'un dossier "background_dir" + d'un CSV (images recadrées).
    - Lit les 4 labels d'experts (ex: [4,4,4,2]) et fusionne en 1 label (majoritaire).
    - Renvoie un batch (X, Y) avec X.shape = (batch_size, 224,224,3), Y.shape=(batch_size,).
    """

    def __init__(self, 
                 background_dir,  # dossier d'images "background" (optionnel)
                 data_dir,        # dossier d'images du csv
                 csv_path,        # chemin du csv (avec colonnes: img_name, label1..4)
                 batch_size=8, 
                 shuffle=True, 
                 max_background=300):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = []

        # 1) Charger un sous-ensemble d'images de fond (optionnel)
        self._load_background(background_dir, max_background)

        # 2) Charger via le CSV
        self._load_from_csv(data_dir, csv_path)

        # 3) Shuffle initial
        if self.shuffle:
            np.random.shuffle(self.samples)

    def _load_background(self, folder, max_background):
        if not os.path.exists(folder):
            return  # si pas de dossier background
        all_imgs = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        selected_imgs = random.sample(all_imgs, min(max_background, len(all_imgs)))

        for img_name in selected_imgs:
            img_path = os.path.join(folder, img_name)
            txt_path = img_path.replace(".jpg", ".txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path, "r") as f:
                line = f.readline().strip().split()
                # line[0] ex: "4_4_4_2"
                raw_labels = line[0].split("_")
                labels = [int(x) for x in raw_labels if x.isdigit()]  # [4,4,4,2] par ex

            # Fusion (majorité)
            final_label = self.majority_label(labels)
            # Stocker (img_path, label)
            self.samples.append((img_path, final_label))

    def _load_from_csv(self, data_dir, csv_path):
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            labels = [int(row["label1"]), int(row["label2"]), 
                      int(row["label3"]), int(row["label4"])]
            img_name = row["img_name"]
            img_path = os.path.join(data_dir, img_name)
            if os.path.exists(img_path):
                final_label = self.majority_label(labels)
                self.samples.append((img_path, final_label))

    def majority_label(self, labels):
        """
        Renvoie le label majoritaire dans la liste, 
        ou en cas d'égalité, le premier max (à adapter si besoin).
        """
        return max(labels, key=labels.count)

    def __len__(self):
        # nombre de lots (batches) = nb total d'images / batch_size (arrondi haut)
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, index):
        # Extraire l'intervalle [start, end)
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.samples))

        batch_samples = self.samples[start:end]

        X_batch = []
        Y_batch = []

        for img_path, label in batch_samples:
            # Charger l'image, la convertir en RGB, la resize
            img = Image.open(img_path).convert("RGB")
            img = img.resize((512, 512))
            img_np = np.array(img).astype("float32") / 255.0

            X_batch.append(img_np)
            Y_batch.append(label)

        X_batch = np.stack(X_batch, axis=0)  # shape = (batch_size, 224,224,3)
        Y_batch = np.array(Y_batch, dtype=np.int32)  # shape = (batch_size,)

        return X_batch, Y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)


train_generator = PerImageBatchGeneratorFromCSV(
    background_dir="../../barrage/grp3/crops/background_patches/",
    data_dir="../../barrage/grp3/crops/raw_crops/",
    csv_path="../../barrage/grp3/crops/raw_crops/labels.csv"
)


# ===========================
# 1. Charger le modèle ResNet50 entraîné (.h5)
# ===========================
# Ce modèle doit être entraîné pour 9 classes (avec Dense(9, activation='softmax'))
resnet_full = load_model("resnet50_9classes_uniquelabel.h5")

# Créer un extracteur de features en coupant juste après la couche GlobalAveragePooling2D.
# Ici on suppose que la couche GAP est la troisième dernière couche.
resnet_feat_model = Model(
    inputs=resnet_full.input,
    outputs=resnet_full.layers[1].output  # Par exemple, la sortie du GAP, de dimension (None, 2048)
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
IMG_SIZE = (512, 512)
features_test = []
ids = []

for img_name in os.listdir(TEST_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(TEST_DIR, img_name)
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_np = np.array(img).astype("float32") / 255.0
    img_np = np.expand_dims(img_np, axis=0)  # (1, 512, 512, 3)

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
df_preds.to_csv("xgb_predictions_uniquelabel.csv", index=False)

print("✅ Prédictions terminées avec ResNet50 + XGBoost")
print(df_preds.head())

