import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image
import albumentations as A
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, PReLU
from tensorflow.keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
n = '4'

os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 

df = pd.read_csv("/home/miashs3/data/grp3/crops/raw_crops/labels.csv")
df["major_label"] = df[['label1', 'label2', 'label3', 'label4']].mode(axis=1)[0]

# Garder seulement les lignes avec labels concordants
concordant_df = df[(df[['label1','label2','label3','label4']].nunique(axis=1)) == 1]

train_df, val_df = train_test_split(concordant_df, test_size=0.2, stratify=concordant_df["major_label"], random_state=42)

print(train_df.shape, val_df.shape)


class CustomGenerator(Sequence):
    def __init__(self, df, img_folder, batch_size=16, img_size=(224,224), augment=False):
        self.df = df
        self.img_folder = img_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.4)
        ]) if augment else None

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.img_folder, row["img_name"])
            img = np.array(Image.open(img_path).convert('RGB').resize(self.img_size))
            
            if self.transform:
                img = self.transform(image=img)['image']
            
            img = img / 255.0
            X.append(img)
            y.append(row["major_label"])

        return np.array(X), np.array(y).astype('int')

# Générateurs :
train_gen = CustomGenerator(train_df, "/home/miashs3/data/grp3/crops/raw_crops/", augment=True)
val_gen = CustomGenerator(val_df, "/home/miashs3/data/grp3/crops/raw_crops/")

num_classes = concordant_df["major_label"].nunique()
print(num_classes)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256)(x)
x = PReLU()(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = PReLU()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    verbose=1
)

model.save("cnn_resnet_focal.h5")

model2 = load_model


# =============================================================================
# Extraction des features et entraînement d'un XGBoost pour optimiser le F1 score
# =============================================================================
# On peut utiliser le modèle entraîné pour extraire les features (par exemple la sortie de la base ResNet50)
feature_extractor = Model(inputs=model.input, outputs=base_model(model.input, training=False))

def extract_features(generator):
    features = []
    labels = []
    for X, y in generator:
        feat = feature_extractor.predict(X, verbose=0)
        features.append(feat)
        labels.append(y)
    return np.concatenate(features), np.concatenate(labels)

X_train_feat, y_train_feat = extract_features(train_gen)
X_val_feat, y_val_feat = extract_features(val_gen)

import xgboost as xgb
from sklearn.metrics import f1_score

# Entraînement de XGBoost
xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=len(classes), eval_metric="mlogloss", use_label_encoder=False, n_jobs= 4)
xgb_clf.fit(X_train_feat, y_train_feat)

# Prédictions et calcul du F1 score sur la validation
y_val_pred = xgb_clf.predict(X_val_feat)
f1 = f1_score(y_val_feat, y_val_pred, average='weighted')
print("Weighted F1 Score sur validation :", f1)