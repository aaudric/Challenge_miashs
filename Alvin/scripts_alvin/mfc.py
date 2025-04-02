import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import random 
from PIL import Image
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

class RaggedGenerator(keras.utils.Sequence) :
    def __init__(self, batch_size=32) :
        self.batch_size=batch_size
        self.max_background = 300
        self.backgrounds = []
        self.images = []
        self.labels = []
       
        #self._load_from_csv("/home/miashs3/data/grp3/crops/raw_crops/", "reassigned_labels.csv")
        self._load_from_csv("/home/barrage/grp3/crops/raw_crops/", "reassigned_labels.csv")
        #self.load_test_images("/home/miashs3/data/grp3/datatest/")
        self.load_test_images("/home/barrage/grp3/datatest/")
        print("images loaded")
        self.images = np.array(self.images)[:100]
        self.labels = np.array(self.labels)[:100]
        self.sampling_weights = np.array([1 - self.class_freqs[np.argmax(self.labels[i])]/len(self.images) for i in range(len(self.labels))])[:100]
        self.sampling_weights += 10
        print(self.sampling_weights)
        self.sampling_weights /= np.sum(self.sampling_weights) 
        
        print(self.class_freqs)
        self.on_epoch_end()


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


generator = RaggedGenerator(8)





def create_mlp() :
    inp = keras.Input((2048,))
    #inp2 = keras.Input((2,))
    #all_inp = layers.Concatenate(axis=-1)([inp, inp2])
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


def create_emb_prop() :
    inp = keras.Input((2048,))
    x = layers.Dense(1024)(inp)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(512, activation='linear')
    return keras.Model(inp, out)



backbone = keras.applications.ResNet50(input_shape=(600, 600, 3), weights="imagenet", include_top=False, pooling='avg')
mlp = create_mlp()
adv = create_adversarial_mlp()

optimizer1 = keras.optimizers.Adam(1e-4)
optimizer2 = keras.optimizers.Adam(1e-4)


ep_loss1 = keras.metrics.Mean()
ep_loss2 = keras.metrics.Mean()
adv_loss_mean = keras.metrics.Mean()
test_loss = keras.metrics.Mean()
acc = keras.metrics.CategoricalAccuracy()
adv_accuracy = keras.metrics.BinaryAccuracy()

#backbone.load_weights("/home/miashs3/scripts_alvin/models/backbone_600_adv_50_t01.weights.h5")
#mlp.load_weights("/home/miashs3/scripts_alvin/models/mlp_600_adv_50_t01.weights.h5")

backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_600_noadv.weights.h5")
mlp.load_weights("/home/barrage/grp3/models/alvin/mlp_600_noadv.weights.h5")



temp = 0.8
test_images = (generator.test_images / 255.0).astype(np.float16)
for iter_ in range(50) :

    average_reprs = []
    train_images = (generator.images / 255.0).astype(np.float16)
    features = backbone.predict(train_images)
    print(features)
    classes = np.argmax(generator.labels, axis=1)

    for i in range(9) :
        valid_inds = np.where(classes==i)
        #print(valid_inds)
        reprr =  np.mean(features[valid_inds], axis=0)  ## vecteur en 2048 dimensions
        average_reprs.append(reprr)
    average_reprs = np.array(average_reprs)
    print(average_reprs)
    

    for epoch in range(20) :
        ep_loss1.reset_states()
        ep_loss2.reset_states()
        adv_loss_mean.reset_states()
        test_loss.reset_states()
        acc.reset_states()
        adv_accuracy.reset_states()

        
        for batch in generator :

            with tf.GradientTape(persistent=True) as tape:
                features = backbone(batch["train_img"], training=True)

                features_norm = tf.nn.l2_normalize(features, axis=-1)
                average_reprs_norm = tf.nn.l2_normalize(average_reprs, axis=-1)

                cosine_similarity = tf.matmul(features_norm, average_reprs_norm, transpose_b=True)  # (B, N_centroides)

                stabilized_distances = cosine_similarity / temp  # Inversion pour que les plus proches aient une proba plus grande
                #stabilized_distances -= tf.reduce_max(stabilized_distances, axis=-1, keepdims=True)  # Soustraction du max pour éviter l'explosion

                softmax_dist = tf.exp(stabilized_distances) / tf.reduce_sum(tf.exp(stabilized_distances), axis=-1, keepdims=True)  # (B, N_centroides)
                entropy_min_loss = -tf.reduce_mean(tf.reduce_sum(batch["train_labels"] * tf.math.log(softmax_dist + 1e-8), axis=-1))

                predictions = mlp(features, training=True)
                classif_loss = tf.keras.losses.categorical_crossentropy(batch["train_labels"], predictions)



            ep_loss1.update_state(entropy_min_loss)
            ep_loss2.update_state(classif_loss)
            acc.update_state(batch["train_labels"], predictions)
            gradients = tape.gradient(entropy_min_loss, backbone.trainable_variables)
            optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables))
            gradients = tape.gradient(classif_loss, mlp.trainable_variables)
            optimizer1.apply_gradients(zip(gradients, mlp.trainable_variables))
            del tape, gradients, features



            ### LOSS TEST

            test_inds = np.random.choice(np.arange(len(test_images)), replace=False, size=(generator.batch_size))
            sampled_test_img = (test_images[test_inds] /255.0).astype(np.float16)

            with tf.GradientTape() as tape:
                test_features = backbone(sampled_test_img, training=True)
                softlabels = mlp(test_features, training=False)

                features_norm = tf.nn.l2_normalize(test_features, axis=-1)
                average_reprs_norm = tf.nn.l2_normalize(average_reprs, axis=-1)
                cosine_similarity = tf.matmul(features_norm, average_reprs_norm, transpose_b=True)  # (B, N_centroides)

                stabilized_distances = cosine_similarity / temp  # Inversion pour que les plus proches aient une proba plus grande
                #stabilized_distances -= tf.reduce_max(stabilized_distances, axis=-1, keepdims=True)  # Soustraction du max pour éviter l'explosion
                softmax_dist = tf.exp(stabilized_distances) / tf.reduce_sum(tf.exp(stabilized_distances), axis=-1, keepdims=True)  # (B, N_centroides)

                entropy_min_loss_test = -tf.reduce_mean(tf.reduce_sum(softlabels * tf.math.log(softmax_dist + 1e-8), axis=-1))
            test_loss.update_state(entropy_min_loss_test)
            gradients = tape.gradient(entropy_min_loss_test, backbone.trainable_variables)
            optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables))



            #### ADVERSARIAL

            with tf.GradientTape(persistent=True) as tape :
                features1 = backbone(batch["train_img"], training=True)
                features2 = backbone(sampled_test_img, training=True)
                features = tf.concat([features1, features2], axis=0)
                print(features)
                adv_labels = tf.concat([tf.ones((generator.batch_size)), tf.zeros((generator.batch_size))], axis=0)
                adv_pred = adv(features, training=True)
                print(adv_pred)
                adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(adv_labels, tf.expand_dims(adv_pred, axis=1)))
                neg_loss = -adv_loss

            
            gradients = tape.gradient(neg_loss, backbone.trainable_variables)
            optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables))
            gradients = tape.gradient(adv_loss, adv.trainable_variables)
            optimizer2.apply_gradients(zip(gradients, adv.trainable_variables))
            del tape, gradients
            adv_loss_mean.update_state(adv_loss)
            adv_accuracy.update_state(adv_labels, adv_pred)


            print("EPOCH ", epoch, "ENDED  LOSSES:", ep_loss1.result(), ep_loss2.result(), adv_loss_mean.result(), "ACCURACY :", acc.result(), adv_accuracy.result() )

    print("FIN ITERATION", iter_)

    
    #backbone.save_weights("/home/miashs3/scripts_alvin/models/backbone_mfc_"+str(iter_)+".weights.h5")
    #mlp.save_weights("/home/miashs3/scripts_alvin/models/mlp_mfc_"+str(iter_)+".weights.h5")
    backbone.save_weights("/home/barrage/grp3/models/alvin/backbone_mfc_"+str(iter_)+".weights.h5")
    mlp.save_weights("/home/barrage/grp3/models/alvin/mlp_mfc_"+str(iter_)+".weights.h5")
        

backbone.save_weights("/home/barrage/grp3/models/alvin/backbone_mfc.weights.h5")
mlp.save_weights("/home/barrage/grp3/models/alvin/mlp_mfc.weights.h5")

#backbone.save_weights("/home/miashs3/scripts_alvin/models/backbone_mfc.weights.h5")
#mlp.save_weights("/home/miashs3/scripts_alvin/models/mlp_mfc.weights.h5")


toto()

test_names = generator.test_names
test_images = generator.test_images.astype(np.float16) / 255.0
test_ratios = generator.test_dimensions

features_test = backbone.predict(test_images)
labels_test = mlp.predict(features_test)
print(labels_test)

labels_test = tf.argmax(labels_test, axis=1).numpy()



df = pd.DataFrame({"idx": test_names, "gt": labels_test})
df.to_csv("result_adv_600_reas_t01.csv", index=False)










#model = create_model()
#model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1e-4))
#model.fit(generator, epochs=10)