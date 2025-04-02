import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import random 
from PIL import Image
from sklearn.manifold import TSNE
import pandas as pd
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

n = '4'

os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 


class RaggedGenerator(keras.utils.Sequence) :
    def __init__(self, batch_size=32) :
        self.batch_size=batch_size
        self.max_background = 300
        self.backgrounds = []
        self.images = []
        self.labels = []
        #self.only_sure = only_sure
       
        self._load_from_csv("/home/barrage/grp3/crops/raw_crops/", "reassigned_labels.csv")
        self.load_test_images("/home/barrage/grp3/datatest/")
        print("images loaded")
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.sampling_weights = np.array([1 - self.class_freqs[np.argmax(self.labels[i])]/len(self.images) for i in range(len(self.labels))])
        self.sampling_weights /= np.sum(self.sampling_weights) 
        
        print(self.class_freqs)
        self.on_epoch_end()
        print(self.df)
        
    def get_csv(self) :
        return self.df


    def load_test_images(self, folder_path):
   
        names = []
        images_resized = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(".jpg"):
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path)

                img_resized = img.resize((600, 600))
                img_np = np.array(img_resized, dtype=np.uint8)

                images_resized.append(img_np)
                h, w = img_np.shape[:2]
                names.append(file[:-4])

        self.test_names = np.array(names)
        self.test_images = np.array(images_resized)


    def _load_from_csv(self, data_dir, csv_path):
        df = pd.read_csv(data_dir+csv_path)
        class_freqs = {}

        self.df = df
        print(self.df)
        for _, row in df.iterrows():
            label = np.zeros((9))
            label[int(row["reassigned_label1"])] = 1
            img_name = row["img_name"]
            img_path = os.path.join(data_dir, img_name)

            if np.argmax(label) not in class_freqs :
                class_freqs[np.argmax(label)] = 1
            else :
                class_freqs[np.argmax(label)] += 1
            
            if os.path.exists(img_path):
                im = Image.open(img_path).convert("RGB")
                im_np = np.array(im)
                h, w = im_np.shape[:2]

                im = im.resize((600, 600))
                im_np = np.array(im, dtype=np.uint8)
                        
                self.images.append(im_np)
                self.labels.append(label)


           

        self.class_freqs = class_freqs

    

      
    def __len__(self) :
        return len(self.images) // self.batch_size


    def __getitem__(self, idx):
        selected_indices = np.random.choice(len(self.images), self.batch_size, p=self.sampling_weights)

        batch_img = self.images[selected_indices]
        #batch_ratios = self.ratios[selected_indices] + np.random.randint(-10, 10, (self.batch_size, 2)) / 3000
        batch_labels = self.labels[selected_indices]

        # Appliquer les augmentations
        batch_img = tf.image.random_flip_left_right(batch_img)
        batch_img = tf.image.random_flip_up_down(batch_img)
        batch_img = tf.image.random_brightness(batch_img, max_delta=0.1)
        batch_img = tf.image.random_contrast(batch_img, lower=0.9, upper=1.1)
        batch_img = tf.image.random_saturation(batch_img, lower=0.9, upper=1.1)

        # Générer batch test de manière aléatoire
        batch_img_inds = np.random.choice(len(self.test_images), self.batch_size, replace=False)
        test_img = self.test_images[batch_img_inds]
        
        # Augmentations
        test_img = tf.image.random_flip_left_right(test_img)
        test_img = tf.image.random_flip_up_down(test_img)
        test_img = tf.image.random_brightness(test_img, max_delta=0.1)
        test_img = tf.image.random_contrast(test_img, lower=0.9, upper=1.1)
        test_img = tf.image.random_saturation(test_img, lower=0.9, upper=1.1)

        adversarial_labels = tf.expand_dims(tf.concat([tf.ones(self.batch_size), tf.zeros(self.batch_size)], axis=0), axis=1)

        return {
            "train_img": tf.cast(batch_img, dtype=tf.float16) / 255.0,
            #"train_ratio": batch_ratios,
            "train_labels": batch_labels,
            "test_img": tf.cast(test_img, dtype=tf.float16) / 255.0,
            "adv_labels": adversarial_labels
        }

    

    def on_epoch_end(self):
        indices = np.arange(len(self.images))
        random.shuffle(indices)
        self.images = self.images[indices]
        #self.ratios = self.ratios[indices]
        self.labels = self.labels[indices]


generator = RaggedGenerator(4)


import re

def extract_project_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readline().strip()  # Lire la première ligne du fichier

            # Utilisation d'une regex pour extraire le projet
            match = re.search(r'^\d+_\d+_\d+_\d+\s+([\w/]+)', content)
            if match:
                project = match.group(1)

                # Ne garder que la partie avant _ ou / ou espace
                project = re.split(r'[_/\s]', project)[0]
                
                return project
    except FileNotFoundError:
        return None  # Si le fichier n'existe pas



import json
import numpy as np
import pandas as pd

# Charger le JSON
data = {
    "0": {"abdomen": 94, "antenne": 0, "pattes": 0, "tete": 95, "background": 0},
    "1": {"abdomen": 128, "antenne": 0, "pattes": 0, "tete": 90, "background": 0},
    "2": {"abdomen": 58, "antenne": 0, "pattes": 0, "tete": 81, "background": 0},
    "3": {"abdomen": 67, "antenne": 0, "pattes": 0, "tete": 41, "background": 0},
    "4": {"abdomen": 23, "antenne": 0, "pattes": 0, "tete": 127, "background": 0},
    "5": {"abdomen": 76, "antenne": 0, "pattes": 0, "tete": 204, "background": 0},
    "6": {"abdomen": 28, "antenne": 0, "pattes": 0, "tete": 28, "background": 0},
    "7": {"abdomen": 41, "antenne": 0, "pattes": 0, "tete": 67, "background": 0},
    "8": {"abdomen": 26, "antenne": 0, "pattes": 0, "tete": 197, "background": 0}
}

# Transformer en DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Calculer la somme des parties pour chaque classe
df["total"] = df.sum(axis=1)

# Calculer les proportions
df_proportions = df.div(df["total"], axis=0).drop(columns=["total"])  # Normalisation
df_proportions *= 100  # Convertir en pourcentage

# Affichage des résultats
print(df_proportions)





# Appliquer la fonction pour modifier la colonne "project"
df = generator.get_csv()
df["project"] = df["id"].apply(lambda x: extract_project_from_file(f"/home/barrage/grp3/data/{x}.txt"))


backbone = keras.applications.ResNet50(input_shape=(600, 600, 3), weights="imagenet", include_top=False, pooling='avg')
backbone(np.random.random((1, 600, 600, 3)))
backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_last.weights.h5")

img = (generator.images /255.0).astype(np.float16)
features = backbone.predict(img)

tsne = TSNE(n_components=2)
data_tsne = tsne.fit_transform(features)


projects = df["project"].tolist()

project_dict = {}
project_label = []
project_count = 0

for proj in projects :
    if proj not in project_dict :
        project_dict[proj] = project_count
        project_count+=1
    project_label.append(project_dict[proj])




fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))  

scatter1 = axes[0].scatter(data_tsne[:len(generator.images), 0], data_tsne[:len(generator.images), 1], c=np.argmax(generator.labels, axis=1), cmap='tab10', s=30)  # Taille des points ajustée
axes[0].set_title("Projection t-SNE (Labels Maj)")
axes[0].set_xlabel("t-SNE Composant 1")
axes[0].set_ylabel("t-SNE Composant 2")
fig.colorbar(scatter1, ax=axes[0], label="Label")

scatter3 = axes[1].scatter(data_tsne[:len(generator.images), 0], data_tsne[:len(generator.images), 1], c=project_label, cmap='tab10', s=30)  # Taille des points ajustée
axes[1].set_title("Projection t-SNE (Projet)")
axes[1].set_xlabel("t-SNE Composant 1")
axes[1].set_ylabel("t-SNE Composant 2")
fig.colorbar(scatter3, ax=axes[1], label="Projet")


plt.tight_layout()
fig.savefig("backbone_last_noadv.png")
plt.close()







