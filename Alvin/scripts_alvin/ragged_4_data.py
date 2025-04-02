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

#os.environ["OMP_NUM_THREADS"] = n
#os.environ["OPENBLAS_NUM_THREADS"] = n
#os.environ["MKL_NUM_THREADS"] = n
#os.environ["NUMEXPR_NUM_THREADS"] = n
#os.sched_setaffinity(0, {0, 1, 2, 3}) 

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

import tensorflow as tf
import pandas as pd
import os
from PIL import Image
import numpy as np

def load_image(image_path):
    """Charge une image sans modifier sa taille"""
    image = tf.io.read_file(image_path)
    image = tf.expand_dims(tf.io.decode_jpeg(image, channels=3), axis=0)  # Support PNG aussi avec decode_png
    return image

def parse_row(image_path, label):
    """Charge une image et son label"""
    image = load_image(image_path)
    return image, tf.expand_dims(label, axis=0)

def create_dataset(data_dir, csv_path):
    """Crée un tf.data.Dataset avec des images de taille dynamique et labels"""
    df = pd.read_csv(os.path.join(data_dir, csv_path))

    # Création des chemins complets et des labels
    image_paths = [os.path.join(data_dir, fname) for fname in df["img_name"]]
    labels = []

    for _, row in df.iterrows():
        labels_list = [int(row["label1"]), int(row["label2"]), int(row["label3"]), int(row["label4"])]
        label = np.zeros(9, dtype=np.float32)
        for idx in labels_list:
            label[idx] += 1
        
        l = np.zeros((9))
        l[np.argmax(label)] = 1
        labels.append(l)

    # Conversion en TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Chargement dynamique des images
    dataset = dataset.map(parse_row, num_parallel_calls=tf.data.AUTOTUNE)

    # Pas de batching pour garder la taille dynamique
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset





data_dir = "/home/barrage/grp3/crops/raw_crops/"
csv_path = "labels.csv"

dataset = create_dataset(data_dir, csv_path)

# Vérification : itérer sur le dataset
for img, label in dataset.take(20):
    print(f"Image shape: {img.shape}, Label: {label.numpy()}", label.shape)




def create_mlp() :
    inp = keras.Input((256,))
    l1 = layers.Dense(256)(inp)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.3)(l1)
    l1 = layers.Dense(256)(l1)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.3)(l1)
    lout = layers.Dense(9, activation='softmax')(l1)
    return keras.Model(inp, lout)

def create_adversarial_mlp() :
    inp = keras.Input((256,))
    l1 = layers.Dense(256)(inp)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    l1 = layers.Dense(256)(l1)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dropout(0.4)(l1)
    lout = layers.Dense(1, activation='sigmoid')(l1)
    return keras.Model(inp, lout)


def inception_block(input):
    c1 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)
    c2 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)
    c3 = layers.Conv2D(101, activation='relu', kernel_size=1, strides=1, padding='same')(input)

    c4 = layers.Conv2D(156, activation='relu', kernel_size=3, strides=1, padding='same')(c1)
    c5 = layers.Conv2D(156, activation='relu', kernel_size=5, strides=1, padding='same')(c2)
    c6 = layers.AveragePooling2D((2, 2), strides=1, padding='same')(c3)

    c7 = layers.Conv2D(109, activation='relu', kernel_size=1, strides=1, padding='same')(input)

    conc = layers.Concatenate(axis=-1)([c4, c5, c6, c7])   # 156 + 156 + 101 + 109 = 522
    return conc


def basic_backbone() :
    inp = keras.Input((None, None, 3))
    c1 = layers.Conv2D(96, padding='same', strides=1, kernel_size=3)(inp) # 64
    c1 = layers.PReLU(shared_axes=[1, 2])(c1) 
    c1 = layers.Conv2D(96, padding='same', strides=1, kernel_size=3)(c1) # 64
    c1 = layers.PReLU(shared_axes=[1, 2])(c1)
    p1 = layers.AveragePooling2D((2, 2))(c1)  # 32
    c3 = layers.Conv2D(128, padding='same', strides=1, kernel_size=3)(p1)
    c3 = layers.PReLU(shared_axes=[1, 2])(c3)
    c4 = layers.Conv2D(128, padding='same', kernel_size=3, strides=1)(c3)  #32
    c4 = layers.PReLU(name='c4', shared_axes=[1, 2])(c4) 
    p2 = layers.AveragePooling2D((2, 2))(c4)  # 16
    c5 = layers.Conv2D(256, padding='same', strides=1, kernel_size=3)(p2) #16
    c5 = layers.PReLU(shared_axes=[1, 2])(c5)

    c6 = layers.Conv2D(256, padding='same', kernel_size=3, strides=1)(c5)  #16
    c6 = layers.PReLU(shared_axes=[1, 2])(c6)
    p3 = layers.AveragePooling2D((2, 2))(c6) # 8
    c7 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(p3) # 6
    c7 = layers.PReLU(shared_axes=[1, 2])(c7)
    c8 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c7) # 4
    c8 = layers.PReLU(shared_axes=[1, 2])(c8)
    c9 = layers.Conv2D(256, padding='same', kernel_size=3, strides=1)(c8) # 2, 2, 256
    c9 = layers.PReLU(shared_axes=[1, 2])(c9)
    flat = layers.GlobalAveragePooling2D(name='flatten')(c9) # 2, 2, 256 = 1024 

    l1 = layers.Dense(1024)(flat) 
    l1 = layers.PReLU()(l1)
    l2 = layers.Dense(1024)(l1)
    l2 = layers.PReLU()(l2)
    
    lout = layers.Dense(9, activation='softmax')(l2)
   
    return keras.Model(inputs=inp, outputs=lout)

batch_size = 16


model = basic_backbone()
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1e-4), run_eagerly=True)

model.fit(dataset, epochs=10)



import time 
t0 = time.time()


class MultiModel(keras.Model) :
    def __init__(self) :
        super(MultiModel, self).__init__()
        self.backbone = basic_backbone(16)
        self.dense = create_mlp()
        self.adv = create_adversarial_mlp()
        self.optimizer1 = keras.optimizers.Adam(1e-4)
        self.optimizer2 = keras.optimizers.Adam(1e-4)

    
    #@tf.function(input_signature=[[tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.float32) for _ in range(16)]])
    def call(self, inputs, training=True) :
        #print("inpt", inputs)
        #x, y = inputs
        features = self.backbone(inputs, training=training)  #liste [1, 512
        #print("features fin back")
        features = layers.Concatenate(axis=0)(features)
        #print("features concat")
        #features = tf.squeeze(tf.concat(features, axis=0), axis=1)
        #print("features concated", features)
        pred = self.dense(features, training=training)
        #print("pred")
        return pred


    def train_step(self, data) :
        train_img, train_labels = data
        #print("j'ai unzip data")
        with tf.GradientTape() as tape :
            features = self.backbone(train_img, training=True)
            features = layers.Concatenate(axis=0)(features)
            pred = self.dense(features, training=True)
            loss = tf.keras.losses.categorical_crossentropy(train_labels, pred)
        gradients = tape.gradient(loss, self.backbone.trainable_variables + self.dense.trainable_variables)
        self.optimizer1.apply_gradients(zip(gradients, self.backbone.trainable_variables + self.dense.trainable_variables))

        """
        print("premiere maj ready")
        with tf.GradientTape(persistent=True) as tape :
            features1 = self.backbone(train_img, training=True) 
            features2 = self.backbone(test_img, training = True)
            features1 = layers.Concatenate(axis=0)(features1)
            features2 = layers.Concatenate(axis=0)(features2)
            features = layers.Concatenate(axis=0)([features1, features2])
            adv_pred = self.adv(features, training=True)
            adv_loss = tf.keras.losses.binary_crossentropy(adv_labels, adv_pred)
            neg_loss = -adv_pred
        gradients = tape.gradient(adv_loss, self.adv.trainable_variables)
        self.optimizer2.apply_gradients(zip(gradients, self.adv.trainable_variables))
        gradients = tape.gradient(neg_loss, self.backbone.trainable_variables)
        self.optimizer2.apply_gradients(zip(gradients, self.backbone.trainable_variables))
        """

        return {"classif_loss":loss}

model = MultiModel()
model.compile(run_eagerly=True)

model.fit(generator, epochs=10)



toto()

in_ = [tf.cast(tf.convert_to_tensor(np.random.random((1, random.randint(200, 400), random.randint(200, 400), 3))), dtype=tf.float32) for _ in range(batch_size)]


print(in_[0])
a = model(in_)  
print("a",a)






in_ = [tf.cast(tf.convert_to_tensor(np.random.random((1, random.randint(200, 400), random.randint(200, 400), 3))), dtype=tf.float16) for _ in range(batch_size)]

print(time.time() -t0)


acc = tf.keras.metrics.CategoricalAccuracy()
avg_loss = tf.keras.metrics.Mean()
avg_adv_loss = tf.keras.metrics.Mean()

for epoch in range(100):
    acc.reset_states()
    avg_loss.reset_states()
    avg_adv_loss.reset_states()
    for batch in generator:
        with tf.GradientTape() as tape :
            #outputs = tf.TensorArray(dtype=tf.float16, size=16, dynamic_size=True)
            #for i, img in enumerate(batch["train_img"]) :
            #    print(i)
            #    feat = forward(tf.convert_to_tensor(img, dtype=tf.float16))
            #    outputs = outputs.write(i, feat)
            #outputs = tf.squeeze(outputs.stack(), axis=1)
            outputs = tf.concat(forward(batch["train_img"]), axis=0)    # 
            print(outputs.shape)
            pred = mlp(outputs, training=True)
            loss = tf.keras.losses.categorical_crossentropy(batch["train_labels"], pred)
        acc.update_state(batch["train_labels"], pred)
        avg_loss.update_state(loss)
        gradients = tape.gradient(loss, backbone.trainable_variables + mlp.trainable_variables)
        optimizer1.apply_gradients(zip(gradients, backbone.trainable_variables+mlp.trainable_variables))

        del gradients, outputs, pred

        with tf.GradientTape(persistent=True) as tape :
           
            features1 = tf.concat(forward(batch["train_img"]), axis=0)
            features2 = tf.concat(forward(batch["test_img"]), axis=0)
            features = tf.concat([features1, features2], axis=0)
            pred = adv(features, training=True)
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