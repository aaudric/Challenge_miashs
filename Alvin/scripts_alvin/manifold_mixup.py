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
        self.alpha = 0.2
        self.backgrounds = []
        self.images = []
        self.labels = []
       
        self._load_from_csv("/home/barrage/grp3/crops/raw_crops/", "reassigned_labels.csv")
        self.load_test_images("/home/barrage/grp3/datatest/")
        print("images loaded")
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.sampling_weights = np.array([1 - self.class_freqs[np.argmax(self.labels[i])]/len(self.images) for i in range(len(self.labels))])
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
        selected_indices = np.random.choice(len(self.images), self.batch_size*2, p=self.sampling_weights, replace=True)

        batch_img = self.images[selected_indices]
        batch_labels = self.labels[selected_indices]   

        batch_img = tf.image.random_flip_left_right(batch_img)
        batch_img = tf.image.random_flip_up_down(batch_img)
        batch_img = tf.image.random_brightness(batch_img, max_delta=0.1)
        batch_img = tf.image.random_contrast(batch_img, lower=0.9, upper=1.1)
        batch_img = tf.image.random_saturation(batch_img, lower=0.9, upper=1.1)
        #print("avant mix",batch_img.shape)

        #batch_a, batch_b = tf.split(batch_img, 2, 0)   # 2 split sur axe 0
        #y_a, y_b = tf.split(batch_labels, 2, 0)


        #lam=np.random.beta(self.alpha, self.alpha, size=self.batch_size)
        #batch_img = lam[:, np.newaxis, np.newaxis, np.newaxis] * batch_a + (1.-lam[:,np.newaxis, np.newaxis, np.newaxis]) * batch_b
        #batch_labels = lam[:, np.newaxis] * y_a + (1.-lam[:, np.newaxis])*y_b
        #print("apres mixup",batch_img.shape, batch_labels.shape)

        batch_img_inds = np.random.choice(len(self.test_images), self.batch_size, replace=False)
        test_img = self.test_images[batch_img_inds]
        
        test_img = tf.image.random_flip_left_right(test_img)
        test_img = tf.image.random_flip_up_down(test_img)
        test_img = tf.image.random_brightness(test_img, max_delta=0.1)
        test_img = tf.image.random_contrast(test_img, lower=0.9, upper=1.1)
        test_img = tf.image.random_saturation(test_img, lower=0.9, upper=1.1)
        #print("test", test_img.shape)


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
mlp = create_mlp()
adv = create_adversarial_mlp()

optimizer1 = keras.optimizers.Adam(1e-4)
optimizer2 = keras.optimizers.Adam(1e-4)


ep_loss = keras.metrics.Mean()
adv_loss = keras.metrics.Mean()
acc = keras.metrics.SparseCategoricalAccuracy()
adv_accuracy = keras.metrics.BinaryAccuracy()

#backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_600_adv_59_t01.weights.h5")
#mlp.load_weights("/home/barrage/grp3/models/alvin/mlp_600_adv_t01.weights.h5")

#adv.load_weights("/home/barrage/grp3/models/alvin/adv_mlp.weights.h5")






temp = 1
for epoch in range(80) :
    acc.reset_states()
    adv_loss.reset_states()
    ep_loss.reset_states()
    adv_accuracy.reset_states()
    for batch in generator :
        
        with tf.GradientTape() as tape :
            features1 = tf.cast(backbone(batch["train_img"][:generator.batch_size], training=True), dtype=tf.float32)
            features2 = tf.cast(backbone(batch["train_img"][generator.batch_size:], training=True), dtype=tf.float32)
            #predictions = mlp([features, batch["train_ratio"]], training=True)
            lam=np.random.beta(generator.alpha, generator.alpha, size=generator.batch_size)
            features = lam[:, tf.newaxis, tf.newaxis, tf.newaxis] * features1 + (1.-lam[:,tf.newaxis, tf.newaxis, tf.newaxis]) * features2
            softlabels = lam[:, tf.newaxis] * batch["train_labels"] + (1.-lam[:, tf.newaxis])*batch["train_labels"]

            logits = tf.cast(mlp(features, training=True), dtype=tf.float32)
            logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
            #print(logits, tf.reduce_min(logits), tf.reduce_max(logits))
            predictions = tf.cast(tf.nn.softmax( logits / temp ), dtype=tf.float32)
            #print("pred",tf.reduce_any(tf.math.is_nan(predictions)))
            loss_classif = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(softlabels, predictions))
            
        gradients = tape.gradient(loss_classif, backbone.trainable_variables+mlp.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables+mlp.trainable_variables))
        acc.update_state(tf.argmax(softlabels), predictions)
        ep_loss.update_state(loss_classif)
        del gradients, features, predictions

        with tf.GradientTape(persistent=True) as tape :
            features1 = backbone(batch["train_img"], training=True)
            features2 = backbone(batch["test_img"], training=True)
            features = tf.cast(tf.concat([features1, features2], axis=0), dtype=tf.float32)
            adv_pred = tf.cast(adv(features, training=True), dtype=tf.float32)
            #print("adv", tf.reduce_any(tf.math.is_nan(adv_pred)))
            loss_adv = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch["adv_labels"], adv_pred))
            neg_loss_adv = -loss_adv

        adv_loss.update_state(loss_adv)
        adv_accuracy.update_state(batch["adv_labels"], adv_pred)
        gradients = tape.gradient(loss_adv, adv.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, adv.trainable_variables))
        gradients = tape.gradient(neg_loss_adv, backbone.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, backbone.trainable_variables))

        #print("LOSS CLASSIF :", loss_classif, "      ADVERSARIAL :", loss_adv)

    print("EPOCH ", epoch, "ENDED :", ep_loss.result(), acc.result(), adv_loss.result(), "ADV ACCURACY :", adv_accuracy.result() )
    if (epoch+1) % 10 == 0 :
        backbone.save_weights("/home/barrage/grp3/models/alvin/backbone_manifoldmixup_"+str(epoch+50)+".weights.h5")
        mlp.save_weights("/home/barrage/grp3/models/alvin/mlp_manifoldmixup_"+str(50)+".weights.h5")
        print("JE SAVE")
        if epoch == 10 :
            temp = 0.7
        if epoch == 20 :
            temp = 0.65
        if epoch == 40 :
            temp = 0.5
        if epoch == 60 :
            temp = 0.4

    generator.on_epoch_end()


backbone.save_weights("/home/barrage/grp3/models/alvin/backbone_manifoldmixup.weights.h5")
mlp.save_weights("/home/barrage/grp3/models/alvin/mlp_manifoldmixup.weights.h5")





#backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_600_adv.weights.h5")
#mlp.load_weights("/home/barrage/grp3/models/alvin/mlp_600_adv.weights.h5")



test_names = generator.test_names
test_images = generator.test_images.astype(np.float16) / 255.0

features_test = backbone.predict(test_images)
labels_test = mlp.predict(features_test)
print(labels_test)

labels_test = tf.argmax(labels_test, axis=1).numpy()



df = pd.DataFrame({"idx": test_names, "gt": labels_test})
df.to_csv("result_manifoldmixup.csv", index=False)










#model = create_model()
#model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1e-4))
#model.fit(generator, epochs=10)