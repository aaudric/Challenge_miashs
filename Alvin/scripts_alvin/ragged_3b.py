import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import random 
from tensorflow.keras import Input, Model
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
    def __init__(self, batch_size=16, reduce_factor=2, only_sure=True) :
        self.batch_size=batch_size
        self.max_background = 300
        self.reduce_factor = reduce_factor
        self.only_sure = only_sure
        self.backgrounds = []
        self.images = []
        self.ratios = []
        self.labels = []
       
        self._load_from_csv("/home/barrage/grp3/crops/raw_crops/", "labels.csv")
        self.load_test_images("/home/barrage/grp3/datatest/")
        print("images loaded")
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
                img_np = np.array(img, dtype=np.uint8)
                h, w = img_np.shape[:2]
                if self.reduce_factor != 1 :
                    new_h = int(h / self.reduce_factor)
                    new_w = int(w / self.reduce_factor)
                    img = img.resize((new_w, new_h), Image.BILINEAR)
                    img_np = np.array(img, dtype=np.uint8)

                images_resized.append(img_np)
                names.append(file[:-4])

        self.test_names = np.array(names)
        self.test_images = images_resized  


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
                if np.any(label==4) :  # si ils sont tous d'accord
                    
                    if os.path.exists(img_path):
                        im = Image.open(img_path).convert("RGB")
                        im_np = np.array(im)
                        h, w = im_np.shape[:2]
                        if self.reduce_factor != 1 :
                            new_h = int(h / self.reduce_factor)
                            new_w = int(w / self.reduce_factor)
                            im = im.resize((new_w, new_h), Image.BILINEAR)

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
                    if self.reduce_factor != 1 :
                        new_h = int(h / self.reduce_factor)
                        new_w = int(w / self.reduce_factor)
                        im = im.resize((new_w, new_h), Image.BILINEAR)

                    im_np = np.array(im, dtype=np.uint8)
                        
                    self.images.append(im_np)
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

        batch_imgs = []
        for ind in selected_indices :
            im = tf.expand_dims(tf.cast(self.images[ind], dtype=tf.float16), axis=0) / 255.0
            im = tf.image.random_flip_left_right(im)
            im = tf.image.random_flip_up_down(im)
            im = tf.image.random_brightness(im, max_delta=0.1)
            #im = tf.image.random_contrast(im, lower=0.85, upper=1.15)
            #im = tf.image.random_saturation(im, lower=0.85, upper=1.15)

            batch_imgs.append(im)

        
        batch_labels = self.labels[selected_indices]

       
        batch_img_inds = np.random.choice(len(self.test_images), self.batch_size, replace=False)
        test_imgs = []
        for ind in batch_img_inds :
            im = tf.expand_dims(tf.cast(self.test_images[ind], dtype=tf.float16), axis=0) /255.0
            im = tf.image.random_flip_left_right(im)
            im = tf.image.random_flip_up_down(im)
            im = tf.image.random_brightness(im, max_delta=0.1)
            #im = tf.image.random_contrast(im, lower=0.85, upper=1.15)
            #im = tf.image.random_saturation(im, lower=0.85, upper=1.15)
            test_imgs.append(im)

        

        adversarial_labels = tf.expand_dims(tf.concat([tf.ones(self.batch_size), tf.zeros(self.batch_size)], axis=0), axis=1)

        return {
            "train_img": batch_imgs,
            "train_labels": batch_labels,
            "test_img": test_imgs,
            "adv_labels": adversarial_labels
        }

   

    def on_epoch_end(self):
        indices = np.arange(len(self.images))
        #random.shuffle(indices)
        #self.images = self.images[indices]
        #self.ratios = self.ratios[indices]
        #self.labels = self.labels[indices]


generator = RaggedGenerator(16)


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
    l1 = layers.Dense(256)(inp)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    l1 = layers.Dense(256)(l1)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    lout = layers.Dense(1, activation='sigmoid')(l1)
    return keras.Model(inp, lout)


batch_size = 16


backbone = keras.applications.ResNet50(weights="imagenet", include_top=False, pooling='avg')
inputs = [Input(shape=(None, None, 3)) for _ in range(batch_size)]
outputs = [backbone(inp) for inp in inputs]
model = Model(inputs=inputs, outputs=outputs)
mlp = create_mlp()
adv = create_adversarial_mlp()

model = Model(inputs=inputs, outputs=outputs)
mlp = create_mlp()
adv = create_adversarial_mlp()

optimizer1 = keras.optimizers.Adam(1e-4)
optimizer2 = keras.optimizers.Adam(1e-4)



"""
@tf.function(input_signature=[[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float16) for _ in range(batch_size)]])
def forward(inputs) :
    return model(inputs, training=True)
"""

@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float16)])
def forward(inputs) :
  return backbone(inputs, training=True)




acc = tf.keras.metrics.CategoricalAccuracy()
avg_loss = tf.keras.metrics.Mean()
avg_adv_loss = tf.keras.metrics.Mean()

for epoch in range(100):
    acc.reset_states()
    avg_loss.reset_states()
    avg_adv_loss.reset_states()
    for batch in generator:
        with tf.GradientTape() as tape :
            outputs = tf.TensorArray(dtype=tf.float16, size=16, dynamic_size=True)
            for i, img in enumerate(batch["train_img"]) :
                print(i)
                feat = forward(tf.convert_to_tensor(img, dtype=tf.float16))
                outputs = outputs.write(i, feat)
            outputs = tf.squeeze(outputs.stack(), axis=1)
            pred = mlp(outputs, training=True)
            loss = tf.keras.losses.categorical_crossentropy(batch["train_labels"], pred)
        acc.update_state(batch["train_labels"], pred)
        avg_loss.update_state(loss)
        gradients = tape.gradient(loss, backbone.trainable_variables + mlp.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables+mlp.trainable_variables))

        del gradients, outputs, pred

        with tf.GradientTape(persistent=True) as tape :
            outputs = tf.TensorArray(dtype=tf.float16, size=32, dynamic_size=True)
            count=0
            for i, img in enumerate(batch["train_img"]) :
                feat = forward(tf.convert_to_tensor(img, dtype=tf.float16))
                outputs = outputs.write(count, feat)
                count+=1
            for i, img in enumerate(batch["test_img"]) :
                feat = forward(tf.convert_to_tensor(img, dtype=tf.float16))
                outputs = outputs.write(count, feat)
                count+=1

            outputs = tf.squeeze(outputs.stack(), axis=1)
            #features1 = tf.concat(forward(batch["train_img"]), axis=0)
            #features2 = tf.concat(forward(batch["test_img"]), axis=0)
            #features = tf.concat([features1, features2], axis=0)
            pred = adv(outputs, training=True)
            adv_loss = tf.keras.losses.binary_crossentropy(batch["test_labels"], pred)
            neg_loss = -adv_loss

        avg_adv_loss.update_state(adv_loss)
        gradients = tape.gradient(adv_loss, adv.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, adv.trainable_variables))
        gradients = tape.gradient(neg_loss, backbone.trainable_variables)
        optimizer2.apply_gradients(zip(gradients, backbone.trainable_variables))

        del tape, outputs, pred


        print("batch loss :", avg_loss.result(),  "           adv_loss :", avg_adv_loss.result(),  "          accuracy train :", acc.result())


    if epoch % 10 == 0 and epoch > 0:
        backbone.save_weights(f"backbone_original_size_weighted_sampling_{epoch}.weights.h5")
        mlp.save_weights(f"mlp_original_size_weighted_sampling_{epoch}.weights.h5")
        tf.print("JE SAVE")


backbone.save_weights("backbone_original_size_weighted_sampling.weights.h5")
mlp.save_weights("mlp_original_size_weighted_sampling.weights.h5")



























for epoch in range(40) :

    for batch in generator :
        
        with tf.GradientTape() as tape :
            features = tf.concat(backbone(batch["train_img"], training=True), axis=0)  # [1, N, N, 3 ; ...]
            predictions = mlp(features, training=True)  # B, 2048
            loss_classif = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(batch["train_labels"], predictions))
        gradients = tape.gradient(loss_classif, backbone.trainable_variables+mlp.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables+mlp.trainable_variables))
        del gradients, features, predictions

        
        with tf.GradientTape(persistent=True) as tape :
            features1 = tf.concat(backbone(batch["train_img"], training=True), axis=0)   # B, 2048
            features2 = tf.concat(backbone(batch["test_img"], training=True), axis=0)
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
"""