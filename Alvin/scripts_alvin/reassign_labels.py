import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from scipy.spatial.distance import cdist
import tensorflow.keras as keras
import random

from scipy.spatial.distance import cdist
from PIL import Image
import pandas as pd
import os
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
        self.labels = []
        self.only_sure = only_sure
       
        self._load_from_csv("/home/barrage/grp3/crops/raw_crops/", "labels.csv")
        self.load_test_images("/home/barrage/grp3/datatest/")
        print("images loaded")
        self.images = np.array(self.images)
        self.labels_surs = np.array(self.labels_surs)
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
            "train_labels": batch_labels,
            "test_img": tf.cast(test_img, dtype=tf.float16) / 255.0,
            "adv_labels": adversarial_labels
        }

    def on_epoch_end(self):
        indices = np.arange(len(self.images))
        random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]

generator = RaggedGenerator()

def create_mlp() :
    inp = keras.Input((2048,))
    l1 = layers.Dense(1024)(inp)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    l1 = layers.Dense(1024)(l1)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    lout = layers.Dense(9, activation='softmax')(l1)
    return keras.Model(inp, lout)


backbone = keras.applications.ResNet50(input_shape=(600, 600, 3), weights="imagenet", include_top=False, pooling='avg')
backbone(np.random.random((4, 600, 600, 3)))
backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_600_adv_50.weights.h5")
mlp = create_mlp()
mlp(np.random.random((4, 2048)))
mlp.load_weights("/home/barrage/grp3/models/alvin/mlp_600_adv_50.weights.h5")




train_imgs = generator.images.astype(np.float16) / 255.0
labels_surs = generator.labels_surs
labels = np.argmax(generator.labels, axis=1)
print("labels_surs :", labels_surs)

classes = np.argmax(generator.labels, axis=1)
print("classes :", classes)


features = backbone.predict(train_imgs)
probas = mlp.predict(features)

df = generator.df


#### ON COMMENCE PAR TRAITER LES LABELS SURS :

names = []
new_labels = []
for i, prob in enumerate(probas) :


    if  labels_surs[i] == 1 :

        if np.argmax(prob) == labels[i] :
            ### PAS DE PROBLEME ON PREDIT LA CLASSE SURE   ==> pire des cas mauvais overfit   sur fausse labellisation
            names.append(df["id"][i])
            new_labels.append(labels[i])   ## on redonne le label


        else :
            ## on ne prédit pas le label sur ==> outlier qu'on a pas overfitté car mal labellisé ?
            names.append(df["id"][i])
            if np.max(prob) > 0.95 :
                new_labels.append(np.argmax(prob))   ## si on est vraiment sur alors 
            else :
                new_labels.append(labels[i])


    else :

        avis = np.zeros((9))
        avis[int(df["label1"][i])] += 1
        avis[int(df["label2"][i])] += 1
        avis[int(df["label3"][i])] += 1
        avis[int(df["label4"][i])] += 1

        if np.any(avis==3) :
            ## on a tout de même une classe majoritaire

            if np.argmax(prob) == np.argmax(avis) :
                names.append(df["id"][i])
                new_labels.append(np.argmax(avis))

            elif np.max(prob) > 0.9 :
                names.append(df["id"][i])
                new_labels.append(np.argmax(prob))

            else :
                best_avis = -1
                max_prob = 0
                for j in range(len(avis)) :
                    if prob[j] > max_prob and avis[j] > 0:
                        best_avis = j
                        max_prob = prob[j]

                names.append(df["id"][i])
                new_labels.append(best_avis)    
             


        elif np.any(avis==2) :
            count_2 = 0
            for j in range(len(avis)) :
                if avis[j] >1 :
                    count_2 += 1



            if count_2 == 1 :
                ## 2 d'accord et les autres pas sur
                if np.argmax(prob) == np.argmax(avis) :
                    names.append(df["id"][i])
                    new_labels.append(np.argmax(avis))

                elif np.max(prob) > 0.7 :
                    names.append(df["id"][i])
                    new_labels.append(np.argmax(prob))

                else :
                    best_avis = -1
                    max_prob = 0
                    for j in range(len(avis)) :
                        if prob[j] > max_prob and avis[j] > 0:
                            best_avis = j
                            max_prob = prob[j]

                    names.append(df["id"][i])
                    new_labels.append(best_avis)    
                    


            else :

                if avis[np.argmax(prob)] == 2:
                    ### si la max prob est une classe désignée par 2 experts
                    names.append(df["id"][i])
                    new_labels.append(np.argmax(avis))

                
                elif np.max(prob) > 0.7 :
                    names.append(df["id"][i])
                    new_labels.append(np.argmax(avis))

                else :
                    best_avis = -1
                    max_prob = 0
                    for j in range(len(avis)) :
                        if prob[j] > max_prob and avis[j] > 0:
                            best_avis = j
                            max_prob = prob[j]

                    names.append(df["id"][i])
                    new_labels.append(best_avis)    



        else :
            ## les 4 ne sont pas d'accord
            if avis[np.argmax(prob)] == 1 :
                names.append(df["id"][i])
                new_labels.append(np.argmax(avis))

            elif np.max(prob) > 0.5 :
                names.append(df["id"][i])
                new_labels.append(np.argmax(prob))

            else :
                best_avis = -1
                max_prob = 0
                for j in range(len(avis)) :
                    if prob[j] > max_prob and avis[j] > 0 :
                        best_avis = j
                        max_prob = prob[j]

                names.append(df["id"][i])
                new_labels.append(best_avis)
                


reassign_df = pd.DataFrame({"id":names, "label":new_labels})
reassign_df.to_csv("reassignation.csv")





            















