import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import scipy
import random 
from PIL import Image
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
n = '4'

os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 

class RaggedGenerator(keras.utils.Sequence) :
    def __init__(self, batch_size=32) :
        self.batch_size=batch_size
        self.max_background = 300
        self.backgrounds = []
        self.images = []
        self.labels = []
        #self.only_sure = only_sure
       
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
        selected_indices = np.random.choice(len(self.images), self.batch_size, p=self.sampling_weights)

        batch_img = self.images[selected_indices]
        #batch_ratios = self.ratios[selected_indices] + np.random.randint(-10, 10, (self.batch_size, 2)) / 3000
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
        #self.ratios = self.ratios[indices]
        self.labels = self.labels[indices]


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

backbone = keras.applications.ResNet50(input_shape=(600, 600, 3), weights="imagenet", include_top=False, pooling='avg')

mlp = create_mlp()
#adv = create_adversarial_mlp()


backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_600_adv_50.weights.h5")
mlp.load_weights("/home/barrage/grp3/models/alvin/mlp_600_adv_50.weights.h5")


generator = RaggedGenerator(4)


images = (generator.test_images / 255.0).astype(np.float16)
features = backbone.predict(images)
pred = np.argmax(mlp.predict(features), axis=1)


clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
print(features.shape, pred.shape)
clf.fit(features, pred)

coefs = clf.coef_  

# Visualiser l'importance des features
import matplotlib.pyplot as plt
plt.figure(figsize=(100, 5))
plt.imshow(coefs, aspect='auto', cmap='coolwarm')
plt.colorbar()
plt.xlabel("Features (2048)")
plt.ylabel("Classes (9)")
plt.title("Impact des features sur les prédictions")
plt.tight_layout()
plt.savefig("relation_features_classes.png")




top_k=5
top_features_per_class = np.argsort(np.abs(coefs), axis=1)[:, -top_k:]

for class_idx in range(coefs.shape[0]):
    top_features = top_features_per_class[class_idx]  
    print(f"Classe {class_idx} - Top {top_k} features: {top_features}")


features_map_extractor = tf.keras.Model(backbone.input, backbone.get_layer("conv5_block3_3_out").output)

for i in range(9) :

    classes_inds = np.where(pred==i)
    classe_img = x_train[classes_inds]
    to_features = top_features_per_class[i]
    features_map = features_map_extractor.predict(classe_img)


    num_features_to_show = 9
    feature_maps_to_plot = features_map[:, :, :num_features_to_show]  # Prend les 9 premières

    # Plot des feature maps
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(feature_maps_to_plot[:, :, i], cmap='viridis')
        ax.axis("off")
        ax.set_title(f"Feature {i}")

    plt.tight_layout()
    plt.savefig("features_map"+str(i+1)+".png")



