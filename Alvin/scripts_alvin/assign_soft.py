import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
import tensorflow.keras as keras
import random
from PIL import Image
from tensorflow.keras import layers
import pandas as pd
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
       
        #self._load_from_csv("/home/miashs3/data/grp3/crops/raw_crops/", "reassigned_labels.csv")
        self._load_from_csv("/home/barrage/grp3/crops/raw_crops/", "reassigned_labels.csv")
        #self.load_test_images("/home/miashs3/data/grp3/datatest/")
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
        i=0
        for file in os.listdir(folder_path):
            if file.lower().endswith(".jpg"):
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path)

                img_resized = img.resize((600, 600))
                img_np = np.array(img_resized, dtype=np.uint8)

                images_resized.append(img_np)
                h, w = img_np.shape[:2]
                names.append(file[:-4])
            i+=1
            #if i > 50 :
            #    break

        self.test_names = np.array(names)
        self.test_images = np.array(images_resized)


    def _load_from_csv(self, data_dir, csv_path):
        df = pd.read_csv(data_dir+csv_path)
        self.df = df
        class_freqs = {}
        i=0
        for _, row in df.iterrows():
            labels = [int(row["label1"]), int(row["label2"]), int(row["label3"]), int(row["label4"])]
            img_name = row["img_name"]
            img_path = os.path.join(data_dir, img_name)
            label = np.zeros((9))
            for idx in labels :
                label[idx] += 1

            
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

                    
                    self.labels_surs.append(1)

            i+=1
            #if i > 50 :
            #    break

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
        softlabels = self.softlabels[batch_img_inds]

        adversarial_labels = tf.expand_dims(tf.concat([tf.ones(self.batch_size), tf.zeros(self.batch_size)], axis=0), axis=1)

        return {
            "train_img": tf.cast(batch_img, dtype=tf.float16) / 255.0,
            "train_labels": batch_labels,
            "test_img": tf.cast(test_img, dtype=tf.float16) / 255.0,
            "adv_labels": adversarial_labels, 
            "softlabels":softlabels
        }

    def on_epoch_end(self):
        indices = np.arange(len(self.images))
        random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]
        #self.softlabels = self.softlabels[indices]

generator = RaggedGenerator(4)


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

def create_adversarial_mlp() :
    inp = keras.Input((2048,))
    l1 = layers.Dense(1024)(inp)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.5)(l1)
    l1 = layers.Dense(1024)(l1)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.5)(l1)
    lout = layers.Dense(1, activation='sigmoid')(l1)
    return keras.Model(inp, lout)


backbone = keras.applications.ResNet50(input_shape=(600, 600, 3), weights="imagenet", include_top=False, pooling='avg')
mlp = create_mlp()
adv = create_adversarial_mlp()


backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_deepcluster_iter60.weights.h5")
mlp.load_weights("/home/barrage/grp3/models/alvin/mlp_deepcluster_iter60.weights.h5")


test_images = (generator.test_images / 255.0).astype(np.float16)



optimizer1 = tf.keras.optimizers.Adam(1e-4)
optimizer2 = tf.keras.optimizers.Adam(1e-4)

track_loss1 = tf.keras.metrics.Mean()
track_acc1 = tf.keras.metrics.CategoricalAccuracy()
track_loss2 = tf.keras.metrics.Mean()
track_acc2 = tf.keras.metrics.BinaryAccuracy()
track_loss3 = tf.keras.metrics.Mean()
track_acc3 = tf.keras.metrics.CategoricalAccuracy()



T = 0.8
for iter_ in range(61, 500)  :


    track_loss1.reset_states()
    track_acc1.reset_states()
    track_loss2.reset_states()
    track_acc2.reset_states()
    track_loss3.reset_states()
    track_acc3.reset_states()

    train_features = backbone.predict((generator.images/255.0).astype(np.float16))
    #print("train features ",train_features)
    centroids = []
    #print(generator.labels)
    classes = np.argmax(generator.labels, axis=1)
    for i in range(9) :
        inds = np.where(classes==i)
        #print("inds",inds)
        centroids.append(np.mean(train_features[inds], axis=0))
        #print(np.mean(train_features[inds], axis=0))

    
    #print(centroids)
    test_features = backbone.predict((generator.test_images/255.0).astype(np.float16))
    dists = np.linalg.norm(test_features[:, np.newaxis, :] - np.expand_dims(np.array(centroids), axis=0), axis=2)   ## N, 1, 2048       1, 9, 2048
    softlabels = np.exp(-dists / T)
    softlabels /= np.sum(softlabels, axis=1, keepdims=True)
    generator.softlabels = softlabels
    del train_features, test_features, dists, centroids

    for batch_id, batch in enumerate(generator) :
        
        if batch_id % 2 == 0 :
            with tf.GradientTape(persistent=True) as tape :
                features1 = backbone(batch["train_img"], training=True)
                features2 = backbone(batch["test_img"], training=True)
                features = tf.concat([features1, features2], axis=0)

                adv_pred = adv(features, training=True)
                loss_adv = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch["adv_labels"], adv_pred))
                neg_loss = -loss_adv
            gradients = tape.gradient(loss_adv, adv.trainable_variables)
            optimizer2.apply_gradients(zip(gradients, adv.trainable_variables))
            gradients = tape.gradient(neg_loss, backbone.trainable_variables)
            optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables))
            track_loss2.update_state(loss_adv)
            track_acc2.update_state(batch["adv_labels"], adv_pred)
            del tape, features, features1, features2, gradients, adv_pred



        with tf.GradientTape() as tape :
            features = backbone(batch["train_img"], training=True)            
            pred = mlp(features, training=True)
            gt = batch["train_labels"]
            #gt = tf.concat([, batch["softlabels"]], axis=0)
            classif_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(gt, pred))
        gradients = tape.gradient(classif_loss, backbone.trainable_variables+mlp.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables+mlp.trainable_variables))
        track_loss1.update_state(classif_loss)
        track_acc1.update_state(gt, pred)
        del features, gradients, pred, gt


        with tf.GradientTape() as tape :
            features = backbone(batch["test_img"], training=True)
            pred = mlp(features, training=True)
            gt = batch["softlabels"]
            classif_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(gt, pred)) * 0.2
        gradients = tape.gradient(classif_loss, backbone.trainable_variables+mlp.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables+mlp.trainable_variables))
        track_loss3.update_state(classif_loss)
        track_acc3.update_state(gt, pred)
        del features, gradients, pred, gt



    print("FIN ITERATION :", iter_, track_loss1.result(), track_acc1.result(), track_loss2.result(), track_acc2.result(), track_loss3.result(), track_acc3.result())


    if iter_ % 2 :#and iter_>0:
        backbone.save_weights("/home/barrage/grp3/models/alvin/backbone_deepcluster_iter"+str(iter_)+".weights.h5")
        mlp.save_weights("/home/barrage/grp3/models/alvin/mlp_deepcluster_iter"+str(iter_)+".weights.h5")

        all_images = np.concatenate([(generator.images/255.0).astype(np.float16), (generator.test_images/255.0).astype(np.float16)], axis=0)
        train_v_test_labels = np.concatenate([np.ones(len(generator.images)), np.zeros((len(generator.test_images)))], axis=0)
        all_features = backbone.predict(all_images)
        del all_images
        tsne =TSNE()
        tsne_data = tsne.fit_transform(all_features)
        del all_features
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))  

        scatter1 = axes[0].scatter(tsne_data[:len(generator.images), 0], tsne_data[:len(generator.images), 1], c=np.argmax(generator.labels, axis=1), cmap='tab10', s=30)  # Taille des points ajustée
        axes[0].set_title("Projection t-SNE (Labels Maj)")
        axes[0].set_xlabel("t-SNE Composant 1")
        axes[0].set_ylabel("t-SNE Composant 2")
        fig.colorbar(scatter1, ax=axes[0], label="Label")

        scatter3 = axes[1].scatter(tsne_data[:, 0], tsne_data[:, 1], c=train_v_test_labels, cmap='viridis', s=30)  # Taille des points ajustée
        axes[1].set_title("Projection t-SNE (Train vs Test)")
        axes[1].set_xlabel("t-SNE Composant 1")
        axes[1].set_ylabel("t-SNE Composant 2")
        fig.colorbar(scatter3, ax=axes[1], label="Train/Test")

        new_labels = np.concatenate([np.argmax(generator.labels, axis=1) ,np.argmax(softlabels, axis=1)], axis=0)
        scatter4 = axes[2].scatter(tsne_data[:, 0], tsne_data[:, 1], c=new_labels, cmap='tab10', s=30)  # Taille des points ajustée
        axes[2].set_title("Projection t-SNE (Labels re-assignés)")
        axes[2].set_xlabel("t-SNE Composant 1")
        axes[2].set_ylabel("t-SNE Composant 2")
        # Ajouter une légende pour la première figure
        fig.colorbar(scatter4, ax=axes[2], label="Label")
        plt.tight_layout()
        fig.savefig("/home/ch_miashs_3/scripts_alvin/tsne_deep_cluster_"+str(iter_)+".png")
        plt.close()
        print("model saved et tsne plot")
        T = max(0.2, T*0.995)




backbone.save_weights("/home/barrage/grp3/models/alvin/backbone_deepcluster.weights.h5")
mlp.save_weights("/home/barrage/grp3/models/alvin/mlp_deepcluster.weights.h5")