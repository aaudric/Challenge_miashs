import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import random 
from PIL import Image
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
n = '4'

os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

class RaggedGenerator(keras.utils.Sequence) :
    def __init__(self, batch_size=32) :
        self.batch_size=batch_size
        self.max_background = 300
        self.backgrounds = []
        self.images = []
        self.ratios = []
        self.labels = []
       
        self._load_from_csv("/home/barrage/grp3/crops/raw_crops/", "labels.csv")
        self.load_test_images("/home/barrage/grp3/datatest/")
        print("images loaded")
        self.images = np.array(self.images)
        self.ratios = np.array(self.ratios)
        self.labels = np.array(self.labels)
        self.sampling_weights = np.array([1 - self.class_freqs[np.argmax(self.labels[i])]/len(self.images) for i in range(len(self.labels))])
        self.sampling_weights /= np.sum(self.sampling_weights) 
        #for i in range(400) :
        #    print(self.sampling_weights[i], self.labels[i])
        print(self.class_freqs)
        self.on_epoch_end()


    def load_test_images(self, folder_path):
   
        names = []
        images_resized = []
        original_dimensions = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(".jpg"):
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path)

                img_resized = img.resize((512, 512))
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
        class_freqs = {}

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

                im = im.resize((512, 512))
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
        batch_img = tf.image.random_contrast(batch_img, lower=0.8, upper=1.2)
        batch_img = tf.image.random_saturation(batch_img, lower=0.8, upper=1.2)

        # Générer batch test de manière aléatoire
        batch_img_inds = np.random.choice(len(self.test_images), self.batch_size, replace=False)
        test_img = self.test_images[batch_img_inds]
        
        # Augmentations
        test_img = tf.image.random_flip_left_right(test_img)
        test_img = tf.image.random_flip_up_down(test_img)
        test_img = tf.image.random_brightness(test_img, max_delta=0.1)
        test_img = tf.image.random_contrast(test_img, lower=0.8, upper=1.2)
        test_img = tf.image.random_saturation(test_img, lower=0.8, upper=1.2)

        adversarial_labels = tf.expand_dims(tf.concat([tf.ones(self.batch_size), tf.zeros(self.batch_size)], axis=0), axis=1)

        return {
            "train_img": tf.cast(batch_img, dtype=tf.float16) / 255.0,
            "train_ratio": batch_ratios,
            "train_labels": batch_labels,
            "test_img": tf.cast(test_img, dtype=tf.float16) / 255.0,
            "adv_labels": adversarial_labels
        }

    """
    def __getitem__(self, idx):
        start = idx*self.batch_size 
        stop = (idx+1) * self.batch_size
        batch_img = self.images[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_img = tf.image.random_flip_left_right(batch_img)
        batch_img = tf.image.random_flip_up_down(batch_img)
        batch_img = tf.image.random_brightness(batch_img, max_delta=0.1)
        batch_ratios = self.ratios[idx*self.batch_size : (idx+1)*self.batch_size]   + np.random.randint(-10, 10, (self.batch_size, 2))/3000
        batch_labels = self.labels[idx*self.batch_size : (idx+1)*self.batch_size]


        batch_img_inds = np.arange(len(self.test_images))
        random.shuffle(batch_img_inds)
        batch_img_inds = batch_img_inds[:self.batch_size]
        test_img = self.test_images[batch_img_inds]
        test_img = tf.image.random_flip_left_right(test_img)
        test_img = tf.image.random_flip_up_down(test_img)
        test_img = tf.image.random_brightness(test_img, max_delta=0.1)

        adversarial_labels = tf.expand_dims(tf.concat([tf.ones(self.batch_size), tf.zeros(self.batch_size)], axis=0), axis=1)



        return {"train_img":tf.cast(batch_img, dtype=tf.float16)/255.0, "train_ratio":batch_ratios, "train_labels":batch_labels, "test_img":tf.cast(test_img, dtype=tf.float16)/255.0, "adv_labels":adversarial_labels}
    """

    def on_epoch_end(self):
        indices = np.arange(len(self.images))
        random.shuffle(indices)
        self.images = self.images[indices]
        self.ratios = self.ratios[indices]
        self.labels = self.labels[indices]


generator = RaggedGenerator(10)





def create_mlp() :
    inp = keras.Input((2048,))
    inp2 = keras.Input((2,))
    all_inp = layers.Concatenate(axis=-1)([inp, inp2])
    l1 = layers.Dense(1024)(all_inp)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    l1 = layers.Dense(1024)(l1)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    lout = layers.Dense(9, activation='softmax')(l1)
    return keras.Model([inp, inp2], lout)


def create_adversarial_mlp() :
    inp = keras.Input((2048,))
    l1 = layers.Dense(256)(inp)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    l1 = layers.Dense(256)(l1)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    lout = layers.Dense(1, activation='sigmoid')(l1)
    return keras.Model(inp, lout)


backbone = keras.applications.ResNet50(input_shape=(512, 512, 3), weights="imagenet", include_top=False, pooling='avg')
mlp = create_mlp()
adv = create_adversarial_mlp()

optimizer1 = keras.optimizers.Adam(1e-4)
optimizer2 = keras.optimizers.Adam(1e-4)

"""
for epoch in range(40) :

    for batch in generator :
        
        with tf.GradientTape() as tape :
            features = backbone(batch["train_img"], training=True)
            predictions = mlp([features, batch["train_ratio"]], training=True)
            loss_classif = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(batch["train_labels"], predictions))
        gradients = tape.gradient(loss_classif, backbone.trainable_variables+mlp.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables+mlp.trainable_variables))
        del gradients, features, predictions

        
        with tf.GradientTape(persistent=True) as tape :
            features1 = backbone(batch["train_img"], training=True)
            features2 = backbone(batch["test_img"], training=True)
            features = tf.concat([features1, features2], axis=0)
            adv_pred = adv(features, training=True)
            
            loss_adv = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch["adv_labels"], adv_pred))
            neg_loss_adv = -loss_adv

        gradients = tape.gradient(loss_adv, adv.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, adv.trainable_variables))
        gradients = tape.gradient(neg_loss_adv, backbone.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, backbone.trainable_variables))
        

        print("LOSS CLASSIF :", loss_classif)

    print("EPOCH ", epoch, "ENDED")
    if epoch % 10 == 0 and epoch > 0 :
        backbone.save_weights("backbone_base_weighted_sampling_"+str(epoch)+".weights.h5")
        mlp.save_weights("mlp_base_weighted_sampling_"+str(epoch)+".weights.h5")
        print("JE SAVE")


backbone.save_weights("backbone_base_weighted_sampling.weights.h5")
mlp.save_weights("mlp_base_weighted_sampling.weights.h5")
"""



weights_to_eval = ["base_10", "base_20", "base_30", "base_weighted_sampling_10", "base_weighted_sampling_20", "base_weighted_sampling_30"]

for w in weights_to_eval :
    print("je passe au modèle", w)


    backbone.load_weights("./backbone_"+w+".weights.h5")
    mlp.load_weights("./mlp_"+w+".weights.h5")


    test_names = generator.test_names
    test_images = generator.test_images.astype(np.float16) / 255.0
    test_ratios = generator.test_dimensions

    features_test = backbone.predict(test_images)
    labels_test = mlp.predict([features_test, test_ratios])
    print(labels_test)

    labels_test = tf.argmax(labels_test, axis=1).numpy()



    df = pd.DataFrame({"name": test_names, "label": labels_test})
    df.to_csv("result_"+w+".csv")










#model = create_model()
#model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1e-4))
#model.fit(generator, epochs=10)




        















