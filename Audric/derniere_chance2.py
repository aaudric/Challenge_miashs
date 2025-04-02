import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


n = '4'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)


# Chargement du CSV contenant au moins une colonne 'filename'
df = pd.read_csv("aggregated_labels.csv")

# Définir le chemin du dossier où se trouvent vos images
image_folder = "/home/barrage/grp3/crops/raw_crops/"  # MODIFIEZ ce chemin en fonction de votre configuration

# Créer la colonne image_path en concaténant le dossier et le nom de fichier
df["image_path"] = df["img_name"].apply(lambda x: os.path.join(image_folder, x))

# Sauvegarder le CSV mis à jour (optionnel)
df.to_csv("aggregated_labels.csv", index=False)

print("La colonne image_path a été ajoutée.")



classes = ['0','1', '2', '3', '4','5','6','7','8']

def encode_labels(label_str):
    labels = label_str.split(",") if isinstance(label_str, str) and label_str else []
    return [1 if str(c) in labels else 0 for c in classes]

# Encodage des labels
df['encoded_labels'] = df['aggregated_labels_str'].apply(encode_labels)
labels_array = np.stack(df['encoded_labels'].values)
image_paths = df['image_path'].values

# Division train/validation
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels_array, test_size=0.2, random_state=42
)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalisation
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(lambda path, label: load_image(path, label))
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(lambda path, label: load_image(path, label))
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

print("Datasets prêts pour l'entraînement.")




# Supposons que vous avez déjà préparé train_dataset et val_dataset
# et que train_dataset est un tf.data.Dataset contenant (image, label)

# 1. Création du modèle avec EfficientNetB0 en base
input_tensor = Input(shape=(224, 224, 3))
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=input_tensor)
x = base_model.output
x = GlobalAveragePooling2D()(x)
# La couche finale avec une activation sigmoïde pour le problème multi-label
output = Dense(9, activation="sigmoid")(x)  # Ici, 4 classes (adaptez selon votre problème)
model = Model(inputs=base_model.input, outputs=output)

# 2. Entraînement initial avec le modèle de base gelé
base_model.trainable = False
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("Début de l'entraînement initial...")
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=10)

# 3. Fine-tuning : dégeler certaines couches de la base
base_model.trainable = True
# On garde les premières couches gelées et on ne dégelle que les 20 dernières, par exemple :
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompiler avec un taux d'apprentissage plus bas pour éviter de perturber trop les poids
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

print("Début du fine-tuning...")



def plot_training_history(history, save_path="training_history.png"):
    # Vérifier que les clés existent dans history.history
    keys = history.history.keys()
    if 'loss' not in keys or 'val_loss' not in keys:
        print("Les courbes de loss ne sont pas disponibles dans l'objet history.")
        return
    if 'f1' not in keys or 'val_f1' not in keys:
        print("Les courbes de F1 score ne sont pas disponibles dans l'objet history.")
        return

    # Création de la figure
    plt.figure(figsize=(14, 5))

    # Plot de la Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss entraînement')
    plt.plot(history.history['val_loss'], label='Loss validation')
    plt.title('Évolution de la Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot du F1 score
    plt.subplot(1, 2, 2)
    plt.plot(history.history['f1'], label='F1 Score entraînement')
    plt.plot(history.history['val_f1'], label='F1 Score validation')
    plt.title('Évolution du F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    # Enregistrer la figure avant de l'afficher
    plt.savefig(save_path)
    plt.show()

# Exemple d'utilisation :
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20)
plot_training_history(history, save_path="history.png")