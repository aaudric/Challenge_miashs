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

class PerImage4LabelsBatchGeneratorFromCSV(Sequence):
    def __init__(self, 
                 background_dir,       # dossier contenant des images de classe 8
                 data_dir,             # dossier contenant les autres images (classes 0..8)
                 csv_path,             # CSV avec colonnes (label1, label2, label3, label4, img_name, ...)
                 batch_size=4, 
                 shuffle=True, 
                 max_background=100,   # nombre max d'images background à charger
                 target_count=1000      # nb visé d'images par classe (0..8)
                 ):
        
        self.batch_size = batch_size  # tu mets 4
        self.shuffle = shuffle
        self.max_background = max_background
        self.target_count = target_count

        # Étape 1) On charge tout en brut
        raw_samples = []

        # 1.1) Charger des images background
        raw_samples += self._load_background(background_dir)

        # 1.2) Charger les images à partir du CSV
        raw_samples += self._load_from_csv(data_dir, csv_path)

        # Étape 2) Suréchantillonner pour avoir ~ target_count par classe
        self.samples = self._build_balanced_samples(raw_samples)

        # Étape 3) Mélanger
        if self.shuffle:
            random.shuffle(self.samples)

    def _load_background(self, folder):
        """
        Charge jusqu'à self.max_background images "background" (classe 8).
        On suppose que chaque image .jpg a un .txt du même nom contenant "8_8_8_8" ou similaire.
        Retourne une liste de tuples : [(img_path, [l1,l2,l3,l4]), ...].
        """
        samples = []
        all_imgs = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        selected = random.sample(all_imgs, min(self.max_background, len(all_imgs)))
        
        for img_name in selected:
            img_path = os.path.join(folder, img_name)
            txt_path = img_path.replace(".jpg", ".txt")
            if not os.path.exists(txt_path):
                continue

            with open(txt_path, "r") as f:
                line = f.readline().strip().split()
                # ex: "8_8_8_8" => 4 labels
                label_str = line[0]
                labels = [int(x) for x in label_str.split("_")]
                if len(labels) == 4:
                    samples.append((img_path, labels))

        return samples

    def _load_from_csv(self, data_dir, csv_path):
        """
        Lit le CSV et récupère (img_path, [label1, label2, label3, label4]) pour chaque ligne.
        Retourne une liste de tuples.
        """
        df = pd.read_csv(csv_path)
        samples = []
        for _, row in df.iterrows():
            labels = [
                int(row["label1"]), 
                int(row["label2"]), 
                int(row["label3"]), 
                int(row["label4"])
            ]
            img_name = row["img_name"]
            img_path = os.path.join(data_dir, img_name)
            if os.path.exists(img_path):
                samples.append((img_path, labels))
        return samples

    def _build_balanced_samples(self, raw_samples):
        """
        Naïvement, on veut qu'il y ait au moins "self.target_count" échantillons
        contenant chacune des classes 0..8. On duplique donc (img, labels) pour
        les classes sous-représentées.

        ATTENTION : en multi-label, dupliquer pour la classe X augmente aussi
                    les autres classes de l'image.
        """
        # dict {classe: [liste d'indices]} => indices pointant dans raw_samples
        class_to_indices = {c: [] for c in range(9)}
        
        for idx, (img_path, labels) in enumerate(raw_samples):
            unique_lbls = set(labels)  # supprime les doublons
            for c in unique_lbls:
                if 0 <= c <= 8:
                    class_to_indices[c].append(idx)

        # Constructeur d'un "set" d'indices final => on ajoute tout de base
        final_indices = set(range(len(raw_samples)))

        # Comptons le nb d'images par classe
        def count_images_for_class(c, idx_set):
            # nb d'images qui contiennent la classe c dans cet ensemble
            indices_c = [i for i in idx_set if c in set(raw_samples[i][1])]
            return len(indices_c)

        # Pour chacune des classes, si < target_count => on duplique
        for c in range(9):
            nb_c = count_images_for_class(c, final_indices)
            if nb_c < self.target_count:
                needed = self.target_count - nb_c
                # On prend au hasard dans la liste class_to_indices[c]
                if len(class_to_indices[c]) > 0:
                    duplicates = random.choices(class_to_indices[c], k=needed)
                    for d_idx in duplicates:
                        final_indices.add(d_idx)
                else:
                    # Aucune image ne contient c => impossible
                    pass

        # On transforme en liste
        final_list = list(final_indices)
        # On construit la liste sur-échantillonnée
        balanced_samples = [raw_samples[i] for i in final_list]
        return balanced_samples

    def __len__(self):
        """
        Nombre de "batches". 
        NOTE : Ici, chaque sample = 1 batch (si on veut 1 image * 4 copies).
        => On renvoie le nombre total de samples.
        
        Si, au contraire, tu voulais rassembler *plusieurs* samples dans un batch
        (4 images *différentes* => batch_size=4), alors tu ferais `len(self.samples) // self.batch_size`.
        Mais comme ta logique veut "une image * 4 copies", on fait 1 sample par batch.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Renvoie un batch de shape (4, 224, 224, 3) contenant 4 copies de la même image,
        et un vecteur de labels [label1, label2, label3, label4].
        """
        img_path, labels = self.samples[index]

        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        img_np = np.array(img, dtype=np.float32) / 255.0

        # On duplique 4 fois la même image => shape (4, 224,224,3)
        batch_images = np.stack([img_np]*self.batch_size, axis=0)

        # On renvoie un np.array de shape (4,) ou (4,4) ?
        # Dans ton code d'avant, c'était un vecteur (4,) => [l1, l2, l3, l4].
        # Le plus logique : on envoie (4,) car tu dis "4 labels" pour l'image
        batch_labels = np.array(labels, dtype=np.int32)

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.samples)


train_generator = PerImage4LabelsBatchGeneratorFromCSV(
    background_dir="../../barrage/grp3/crops/background_patches/",
    data_dir="../../barrage/grp3/crops/raw_crops/",
    csv_path="../../barrage/grp3/crops/raw_crops/labels.csv"
)


# ===========================
# 1. Charger le modèle ResNet50 entraîné (.h5)
# ===========================
# Ce modèle doit être entraîné pour 9 classes (avec Dense(9, activation='softmax'))
resnet_full = load_model("resnet50_9classes_equilibre.h5")

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
