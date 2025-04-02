import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import random 
from PIL import Image
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

class RaggedGenerator(keras.utils.Sequence) :
    def __init__(self, batch_size=32, only_sure=True, reduce_factor=2) :
        self.batch_size=batch_size
        self.max_background = 300
        self.target_shape = 448
        self.backgrounds = []
        self.images = []
        self.ratios = []
        self.labels = []
        self.max_w = 0
        self.max_h =0
        self.only_sure = only_sure
        self.reduce_factor = reduce_factor
       
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
        self.max_shape = max(self.max_h, self.max_w)
        self.on_epoch_end()


    def load_test_images(self, folder_path):
   
        names = []
        images_resized = []
        original_dimensions = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(".jpg"):
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path)

                #img_resized = img.resize((600, 600))
                img_np = np.array(img, dtype=np.uint8)
                h, w = img_np.shape[:2]
                self.max_h = max(self.max_h, h//self.reduce_factor)
                self.max_w = max(self.max_w, w//self.reduce_factor)
                img_resized = img.resize((w//self.reduce_factor, h//self.reduce_factor), Image.ANTIALIAS)
                img_np = np.array(img_resized, dtype=np.uint8)

                images_resized.append(img_np)
                h, w = img_np.shape[:2]
                original_dimensions.append(np.array([h/3000, w/3000]))  # Stocke (width, height)
                names.append(file[:-4])



        self.test_names = np.array(names)
        self.test_images = np.array(images_resized)
        print(self.test_images.dtype)
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

            if self.only_sure :

                if np.any(label==4) :

                    if os.path.exists(img_path):
                        img = Image.open(img_path)

                        img_np = np.array(img, dtype=np.uint8)
                        h, w = img_np.shape[:2]
                        self.max_h = max(self.max_h, h//self.reduce_factor)
                        self.max_w = max(self.max_w, w//self.reduce_factor)
                        img_resized = img.resize((w//self.reduce_factor, h//self.reduce_factor), Image.ANTIALIAS)
                        img_np = np.array(img_resized, dtype=np.uint8)
                        self.images.append(img_np)
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
                    img = Image.open(img_path)

                    img_np = np.array(img, dtype=np.uint8)
                    h, w = img_np.shape[:2]
                    self.max_h = max(self.max_h, h//self.reduce_factor)
                    self.max_w = max(self.max_w, w//self.reduce_factor)
                    img_resized = img.resize((w//self.reduce_factor, h//self.reduce_factor), Image.ANTIALIAS)
                    img_np = np.array(img_resized, dtype=np.uint8)
                    
                    self.images.append(img_np)
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

        selected_bg_inds = np.random.choice(len(self.images), self.batch_size*2)

        #batch_img = self.images[selected_indices]
        #batch_same_size = np.zeros((self.batch_size, self.max_shape, self.max_shape, 3))
        batch_same_size = []
        for i in range(self.batch_size) :
            im = self.images[selected_indices[i]]
            h, w = im.shape[:2]

            bg = np.array(Image.fromarray(self.images[selected_bg_inds[i]]).resize((self.max_shape, self.max_shape), Image.BILINEAR))
            y1, y2 = (self.max_shape - h) // 2, (self.max_shape - h) // 2 + h
            x1, x2 = (self.max_shape - w) // 2, (self.max_shape - w) // 2 + w
            bg[y1:y2, x1:x2] = im
            bg = tf.expand_dims(tf.image.resize(tf.convert_to_tensor(bg, dtype=tf.float16), (self.target_shape, self.target_shape), method=tf.image.ResizeMethod.BILINEAR), axis=0)
            batch_same_size.append(bg)
        batch_same_size = tf.concat(batch_same_size, axis=0)
        # Appliquer les augmentations
        batch_img = tf.image.random_flip_left_right(batch_same_size)
        batch_img = tf.image.random_flip_up_down(batch_img)
        batch_img = tf.image.random_brightness(batch_img, max_delta=0.1)
        batch_img = tf.image.random_contrast(batch_img, lower=0.9, upper=1.1)
        batch_img = tf.image.random_saturation(batch_img, lower=0.9, upper=1.1)

        batch_labels = self.labels[selected_indices]



        batch_img_inds = np.random.choice(len(self.test_images), self.batch_size, replace=False)
        batch_same_size = []
        for i in range(self.batch_size) :
            im = self.test_images[batch_img_inds[i]]
            h, w = im.shape[:2]

            bg = np.array(Image.fromarray(self.images[selected_bg_inds[i+self.batch_size]]).resize((self.max_shape, self.max_shape), Image.BILINEAR))
            y1, y2 = (self.max_shape - h) // 2, (self.max_shape - h) // 2 + h
            x1, x2 = (self.max_shape - w) // 2, (self.max_shape - w) // 2 + w
            bg[y1:y2, x1:x2] = im
            bg = tf.expand_dims(tf.image.resize(tf.convert_to_tensor(bg, dtype=tf.float16), (self.target_shape, self.target_shape), method=tf.image.ResizeMethod.BILINEAR), axis=0)
            batch_same_size.append(bg)
        batch_same_size = tf.concat(batch_same_size, axis=0)
        # Augmentations
        test_img = tf.image.random_flip_left_right(batch_same_size)
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
        #self.images = self.images[indices]
        #self.ratios = self.ratios[indices]
        #self.labels = self.labels[indices]


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


backbone = keras.applications.ResNet50(input_shape=(448, 448, 3), weights="imagenet", include_top=False, pooling='avg')
mlp = create_mlp()
adv = create_adversarial_mlp()

optimizer1 = keras.optimizers.Adam(1e-4)
optimizer2 = keras.optimizers.Adam(1e-4)

"""
ep_loss = keras.metrics.Mean()
adv_loss = keras.metrics.Mean()
acc = keras.metrics.CategoricalAccuracy()
for epoch in range(50) :
    acc.reset_states()
    adv_loss.reset_states()
    ep_loss.reset_states()
    for batch in generator :
        
        with tf.GradientTape() as tape :
            features = backbone(batch["train_img"], training=True)
            #predictions = mlp([features, batch["train_ratio"]], training=True)
            predictions = mlp(features, training=True)
            loss_classif = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(batch["train_labels"], predictions))
            
        gradients = tape.gradient(loss_classif, backbone.trainable_variables+mlp.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables+mlp.trainable_variables))
        acc.update_state(batch["train_labels"], predictions)
        ep_loss.update_state(loss_classif)
        del gradients, features, predictions

        with tf.GradientTape(persistent=True) as tape :
            features1 = backbone(batch["train_img"], training=True)
            features2 = backbone(batch["test_img"], training=True)
            features = tf.concat([features1, features2], axis=0)
            adv_pred = adv(features, training=True)
            
            loss_adv = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch["adv_labels"], adv_pred))
            neg_loss_adv = -loss_adv

        adv_loss.update_state(loss_adv)
        gradients = tape.gradient(loss_adv, adv.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, adv.trainable_variables))
        gradients = tape.gradient(neg_loss_adv, backbone.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, backbone.trainable_variables))

        #print("LOSS CLASSIF :", loss_classif, "      ADVERSARIAL :", loss_adv)

    print("EPOCH ", epoch, "ENDED :", ep_loss.result(), acc.result() )
    if epoch % 10 == 0 :
        backbone.save_weights("/home/barrage/grp3/models/alvin/backbone_448_prop_adv_"+str(epoch)+".weights.h5")
        mlp.save_weights("/home/barrage/grp3/models/alvin/mlp_448_prop_adv_"+str(epoch)+".weights.h5")
        print("JE SAVE")


backbone.save_weights("/home/barrage/grp3/models/alvin/backbone_448_prop_adv.weights.h5")
mlp.save_weights("/home/barrage/grp3/models/alvin/mlp_448_prop_adv.weights.h5")
"""




backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_448_prop_adv.weights.h5")
mlp.load_weights("/home/barrage/grp3/models/alvin/mlp_448_prop_adv.weights.h5")



test_names = generator.test_names
#print(generator.test_images)
test_images = generator.test_images

test_resized = []
indices = np.random.choice(np.arange(len(generator.images)), replace=True, size=(len(test_images)))
for i in range(len(test_images)) :
    im = test_images[i]

    bg = generator.images[indices[i]]
    h, w = im.shape[:2]

    bg = np.array(Image.fromarray(generator.images[indices[i]]).resize((generator.max_shape, generator.max_shape), Image.BILINEAR))
    y1, y2 = (generator.max_shape - h) // 2, (generator.max_shape - h) // 2 + h
    x1, x2 = (generator.max_shape - w) // 2, (generator.max_shape - w) // 2 + w
    bg[y1:y2, x1:x2] = im
    bg =tf.image.resize(tf.convert_to_tensor(bg, dtype=tf.float16), (generator.target_shape, generator.target_shape), method=tf.image.ResizeMethod.BILINEAR).numpy()
    test_resized.append(bg)


test_images = np.array(test_resized).astype(np.float16) / 255.0
test_ratios = generator.test_dimensions

features_test = backbone.predict(test_images)
labels_test = mlp.predict(features_test)
print(labels_test)

labels_test = tf.argmax(labels_test, axis=1).numpy()



df = pd.DataFrame({"idx": test_names, "gt": labels_test})
df.to_csv("result_adv_448_prop.csv", index=False)










#model = create_model()
#model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1e-4))
#model.fit(generator, epochs=10)