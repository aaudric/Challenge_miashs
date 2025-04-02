import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import random
from PIL import Image
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
n = '4'

os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3, 4}) 





class RaggedGenerator(keras.utils.Sequence) :
    def __init__(self, batch_size=32, only_sure=False) :
        self.batch_size=batch_size
        self.max_background = 300
        self.backgrounds = []
        self.images = []
        self.labels_surs = []
        self.ratios = []
        self.labels = []
        self.only_sure = only_sure
       
        self._load_from_csv("/home/barrage/grp3/crops/raw_crops/", "reassigned_labels.csv")
        self.load_test_images("/home/barrage/grp3/datatest/")
        print("images loaded")
        self.images = np.array(self.images)
        self.labels_surs = np.array(self.labels_surs)
        self.ratios = np.array(self.ratios)
        self.labels = np.array(self.labels)
        self.sampling_weights = np.array([1 - self.class_freqs[np.argmax(self.labels[i])]/len(self.images) for i in range(len(self.labels))])
        self.sampling_weights /= np.sum(self.sampling_weights) 
        #for i in range(400) :
        #    print(self.sampling_weights[i], self.labels[i])
        print(self.class_freqs)
        #self.on_epoch_end()


    def load_test_images(self, folder_path):
   
        names = []
        images_resized = []
        original_dimensions = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(".jpg"):
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path)

                img_resized = img.resize((600, 600))
                img_np = np.array(img_resized, dtype=np.uint8)

                images_resized.append(img_np)
                h, w = img_np.shape[:2]
                original_dimensions.append(np.array([h/3000, w/3000]))  # Stocke (width, height)
                names.append(file[:-4])

        self.test_names = np.array(names)
        self.test_images = np.array(images_resized)
        self.test_dimensions = np.array(original_dimensions)


    def _load_from_csv(self, data_dir, csv_path):
        df = pd.read_csv(data_dir+csv_path)
        self.df = df
        class_freqs = {}
        for _, row in df.iterrows():
            labels = [int(row["label1"]), int(row["label2"]), int(row["label3"]), int(row["label4"])]
            img_name = row["img_name"]
            img_path = os.path.join(data_dir, img_name)
            label = np.zeros((9))
            for idx in labels :
                label[idx] += 1

            if self.only_sure :

                if np.any(label==4) :

                    if os.path.exists(img_path):
                        im = Image.open(img_path).convert("RGB")
                        im_np = np.array(im)
                        h, w = im_np.shape[:2]

                        im = im.resize((600, 600))
                        im_np = np.array(im, dtype=np.uint8)
                        
                        self.images.append(im_np)
                        self.ratios.append([h/3000, w/3000])
                        la = np.zeros((9))
                        if np.argmax(label) not in class_freqs :
                            class_freqs[np.argmax(label)] = 1
                        else :
                            class_freqs[np.argmax(label)] += 1
                        la[np.argmax(label)] = 1
                        self.labels.append(la)
            else :
                if os.path.exists(img_path):
                    im = Image.open(img_path).convert("RGB")
                    im_np = np.array(im)
                    h, w = im_np.shape[:2]

                    im = im.resize((600, 600))
                    im_np = np.array(im, dtype=np.uint8)
                    
                    self.images.append(im_np)
                    self.ratios.append([h/3000, w/3000])
                    la = np.zeros((9))
                    if np.argmax(label) not in class_freqs :
                        class_freqs[np.argmax(label)] = 1
                    else :
                        class_freqs[np.argmax(label)] += 1
                    la[np.argmax(label)] = 1
                    self.labels.append(la)

                    if np.any(label==4) :
                        self.labels_surs.append(1)
                    else :
                        self.labels_surs.append(0)




        self.class_freqs = class_freqs
 
    def __len__(self) :
        return len(self.images) // self.batch_size


    def __getitem__(self, idx):
        selected_indices = np.random.choice(len(self.images), self.batch_size, p=self.sampling_weights)

        batch_img = self.images[selected_indices]
        batch_ratios = self.ratios[selected_indices] + np.random.randint(-10, 10, (self.batch_size, 2)) / 3000
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
        self.ratios = self.ratios[indices]
        self.labels = self.labels[indices]

generator = RaggedGenerator()


backbone = keras.applications.ResNet50(input_shape=(600, 600, 3), weights="imagenet", include_top=False, pooling='avg')
backbone(np.random.random((4, 600, 600, 3)))
backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_600_adv_59_t01.weights.h5")




images = generator.images.astype(np.float16) /  255.0
labels = generator.labels
print(labels)
label_maj = np.argmax(labels, axis=1)


confiance = generator.labels_surs

"""
label_maj = []
for lab in labels :
    temp_lab = np.zeros((9))
    for l in lab :
        temp_lab[int(l)] += 1
    label_maj.append(np.argmax(temp_lab, axis=1))
label_maj = np.array(label_maj)
"""
"""
from collections import Counter
def agreement_score(label1, label2, label3, label4):
    labels = [label1, label2, label3, label4]
    counts = Counter(labels)  # Compte les occurrences de chaque label
    max_freq = max(counts.values())  # Fréquence du label le plus fréquent

    # Attribution du score selon la fréquence du label majoritaire
    if max_freq == 4:
        return 0  # 4 labels identiques
    elif max_freq == 3:
        return 1  # 3 identiques, 1 différent
    elif len(counts) == 2 and all(v == 2 for v in counts.values()):
        return 2  # 2 groupes de 2 identiques (ex: A, A, B, B)
    elif max_freq == 2:
        return 3  # Uniquement 2 identiques, les 2 autres différents
    else:
        return 4

desaccord_level = []
for lab in labels :
    desaccord_level.append(agreement_score(lab[0], lab[1], lab[2], lab[3]))
desaccord_level = np.array(desaccord_level)
"""

"""
projects = generator.projects
project_id = []
projects_ids = {}
project_counter = 0
for proj in projects :
    if proj not in projects_ids :
        projects_ids[proj] = project_counter
        project_counter+=1
    project_id.append(projects_ids[proj]) 
project_id = np.array(project_id)
"""



test_images = generator.test_images.astype(np.float16)  / 255.0

train_v_test = np.concatenate([np.ones(len(images)), np.zeros(len(test_images))], axis=0)



all_images = np.concatenate([images, test_images], axis=0)
all_inds = np.arange(len(all_images))
train_inds = all_inds[:len(images)]
test_inds = all_inds[len(images):]

features = backbone.predict(all_images)


print("juste avant tsne")

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
tsne_data = tsne.fit_transform(features)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))  # Ajustement de la taille du figure

# Premier graphique avec des points plus visibles
scatter1 = axes[0].scatter(tsne_data[:len(images), 0], tsne_data[:len(images), 1], c=label_maj, cmap='tab10', s=30)  # Taille des points ajustée
axes[0].set_title("Projection t-SNE (Labels Maj)")
axes[0].set_xlabel("t-SNE Composant 1")
axes[0].set_ylabel("t-SNE Composant 2")
# Ajouter une légende pour la première figure
fig.colorbar(scatter1, ax=axes[0], label="Label")


scatter2 = axes[1].scatter(tsne_data[:len(images), 0], tsne_data[:len(images), 1], c=confiance, cmap='viridis', s=30)  # Taille des points ajustée
axes[1].set_title("Projection t-SNE (Labels Maj)")
axes[1].set_xlabel("t-SNE Composant 1")
axes[1].set_ylabel("t-SNE Composant 2")
# Ajouter une légende pour la première figure
fig.colorbar(scatter2, ax=axes[1], label="Total agreement")

# Deuxième graphique avec des points plus visibles
scatter3 = axes[2].scatter(tsne_data[:, 0], tsne_data[:, 1], c=train_v_test, cmap='viridis', s=30)  # Taille des points ajustée
axes[2].set_title("Projection t-SNE (Train vs Test)")
axes[2].set_xlabel("t-SNE Composant 1")
axes[2].set_ylabel("t-SNE Composant 2")
fig.colorbar(scatter3, ax=axes[2], label="Train/Test")


new_labels = generator.df["reassigned_label1"]
scatter4 = axes[3].scatter(tsne_data[:len(images), 0], tsne_data[:len(images), 1], c=new_labels, cmap='tab10', s=30)  # Taille des points ajustée
axes[3].set_title("Projection t-SNE (Labels re-assignés)")
axes[3].set_xlabel("t-SNE Composant 1")
axes[3].set_ylabel("t-SNE Composant 2")
# Ajouter une légende pour la première figure
fig.colorbar(scatter4, ax=axes[3], label="Label")

plt.tight_layout()
fig.savefig("tsne_adv_samesize_t01_50_reassigneds.png")























