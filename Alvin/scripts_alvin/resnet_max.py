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



class HSVGenerator(keras.utils.Sequence) :
    def __init__(self, batch_size=32, only_sure=False, soft=True) :
        self.batch_size=batch_size
        self.max_background = 300
        self.backgrounds = []
        self.images = []
        self.soft = True
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
            label /= np.sum(label) 

            
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
                if self.soft :
                    self.labels.append(label)
                else :
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

        img_hsv = tf.image.rgb_to_hsv(images)
        batch_img = tf.concat([batch_img, img_hsv], axis=-1)
        batch_img = tf.image.random_flip_left_right(batch_img)
        batch_img = tf.image.random_flip_up_down(batch_img)
        batch_img = tf.image.random_brightness(batch_img, max_delta=0.1)
        batch_img = tf.image.random_contrast(batch_img, lower=0.9, upper=1.1)
        batch_img = tf.image.random_saturation(batch_img, lower=0.9, upper=1.1)

        batch_img_inds = np.random.choice(len(self.test_images), self.batch_size, replace=False)
        test_img = self.test_images[batch_img_inds]

        img_hsv = tf.image.rgb_to_hsv(test_img)
        test_img = tf.concat([test_img, img_hsv], axis=-1)
        test_img = tf.image.random_flip_left_right(test_img)
        test_img = tf.image.random_flip_up_down(test_img)
        test_img = tf.image.random_brightness(test_img, max_delta=0.1)
        test_img = tf.image.random_contrast(test_img, lower=0.9, upper=1.1)
        test_img = tf.image.random_saturation(test_img, lower=0.9, upper=1.1)
        #softlabels = self.softlabels[batch_img_inds]

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




generator = RaggedGenerator(8)


def create_mlp() :
    inp = keras.Input((2048,))
    l1 = layers.Dense(1024)(inp)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    l1 = layers.Dense(1024)(l1)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    lout = layers.Dense(9, activation='linear')(l1)
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
conv1 = backbone.layers[1]
weights, bias = conv1.get_weights()  # (7,7,3,64) + biais
new_weights = np.concatenate([weights] * 2, axis=2)  # Duplique les poids sur 6 canaux
new_input = keras.layers.Input(shape=(600, 600, 6))
new_conv1 = keras.layers.Conv2D(64, (7, 7), strides=2, padding='same', weights=[new_weights, bias], trainable=True)(new_input)
x = new_conv1
for layer in backbone.layers[2:]:  # On saute la première couche convolutive
    x = layer(x)
backbone = keras.Model(new_input, x)
mlp = create_mlp()
adv = create_adversarial_mlp()

optimizer1 = keras.optimizers.Adam(1e-4)
optimizer2 = keras.optimizers.Adam(1e-4)


ep_loss = keras.metrics.Mean()
adv_loss = keras.metrics.Mean()
acc = keras.metrics.CategoricalAccuracy()
adv_accuracy = keras.metrics.BinaryAccuracy()

temp = 1.2
for epoch in range(80) :
    acc.reset_states()
    adv_loss.reset_states()
    ep_loss.reset_states()
    adv_accuracy.reset_states()
    for batch in generator :
        
        with tf.GradientTape() as tape :
            features = tf.cast(backbone(batch["train_img"], training=True), dtype=tf.float32)
            logits = tf.cast(mlp(features, training=True), dtype=tf.float32)
            logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
            predictions = tf.cast(tf.nn.softmax( logits / temp ), dtype=tf.float32)
            loss_classif = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(batch["train_labels"], predictions))
            
        gradients = tape.gradient(loss_classif, backbone.trainable_variables+mlp.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables+mlp.trainable_variables))
        acc.update_state(batch["train_labels"], predictions)
        ep_loss.update_state(loss_classif)
        del gradients, features, predictions

        with tf.GradientTape(persistent=True) as tape :
            features1 = backbone(batch["train_img"], training=True)
            features2 = backbone(batch["test_img"], training=True)
            features = tf.cast(tf.concat([features1, features2], axis=0), dtype=tf.float32)
            adv_pred = tf.cast(adv(features, training=True), dtype=tf.float32)
            loss_adv = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch["adv_labels"], adv_pred))
            neg_loss_adv = -loss_adv

        adv_loss.update_state(loss_adv)
        adv_accuracy.update_state(batch["adv_labels"], adv_pred)
        gradients = tape.gradient(loss_adv, adv.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, adv.trainable_variables))
        gradients = tape.gradient(neg_loss_adv, backbone.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, backbone.trainable_variables))



    print("EPOCH ", epoch, "ENDED :", ep_loss.result(), acc.result(), adv_loss.result(), "ADV ACCURACY :", adv_accuracy.result() )
    if (epoch+1) % 10 == 0 :
        backbone.save_weights("/home/barrage/grp3/models/alvin/backbone_hsv_"+str(epoch+50)+".weights.h5")
        mlp.save_weights("/home/barrage/grp3/models/alvin/mlp_hsv_"+str(50)+".weights.h5")
        print("JE SAVE")
        if epoch == 10 :
            temp = 1
        if epoch == 20 :
            temp = 0.9
       

    generator.on_epoch_end()


backbone.save_weights("/home/barrage/grp3/models/alvin/backbone_600_adv_t01.weights.h5")
mlp.save_weights("/home/barrage/grp3/models/alvin/mlp_600_adv_t01.weights.h5")