from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
from PIL import Image

test_images_dir = "data/grp3/datatest/" # dossier contenant vos images test

# Chargement du modèle entraîné
model_final = load_model("model_final.h5", compile=False)

# Fonction pour charger et préparer les images test
def preprocess_image(img_path, img_size=(512,512)):
    img = Image.open(img_path).convert("RGB").resize(img_size)
    img_np = np.array(img) / 255.0
    return np.expand_dims(img_np, axis=0)

# Prédiction sur toutes les images test
results = []

for fname in sorted(os.listdir(test_images_dir)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(test_images_dir, fname)
        img_array = preprocess_image(img_path)
        pred = model_final.predict(img_array, verbose=0)
        pred_label = np.argmax(pred, axis=1)[0]
        results.append({"idx": fname, "gt": pred_label})

# Enregistrer les résultats dans un CSV
predictions_df = pd.DataFrame(results)
predictions_df.to_csv("predictions.csv", index=False)

print(predictions_df.head())