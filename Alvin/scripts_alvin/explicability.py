import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import scipy
import random 
import json
from PIL import Image
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

#from tensorflow.keras.mixed_precision import set_global_policy
#set_global_policy('mixed_float16')
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
        batch_labels = self.labels[selected_indices]

        batch_img = tf.image.random_flip_left_right(batch_img)
        batch_img = tf.image.random_flip_up_down(batch_img)
        batch_img = tf.image.random_brightness(batch_img, max_delta=0.1)
        batch_img = tf.image.random_contrast(batch_img, lower=0.9, upper=1.1)
        batch_img = tf.image.random_saturation(batch_img, lower=0.9, upper=1.1)

        batch_img_inds = np.random.choice(len(self.test_images), self.batch_size, replace=False)
        test_img = self.test_images[batch_img_inds]
        
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
        self.labels = self.labels[indices]



inp = tf.keras.Input((200, 200, 3))
c1 = layers.Conv2D(64, activation='relu', padding='valid', kernel_size=3)(inp)   # 198
c2 = layers.Conv2D(64, activation='relu', padding='valid', kernel_size=3)(c1)   # 196
c3 = layers.Conv2D(128, activation='relu', padding='valid', strides=2, kernel_size=3)(c2)  # 96
c1 = layers.Conv2D(128, activation='relu', padding='valid', kernel_size=3)(c3)   # 94
c2 = layers.Conv2D(256, activation='relu', padding='valid', kernel_size=3)(c1)   # 92
c3 = layers.Conv2D(256, activation='relu', padding='valid', strides=2, kernel_size=3)(c2) # 46
p = layers.GlobalMaxPooling2D()(c3)
l1 = layers.Dense(128, activation='relu')(p)
l1 = layers.Dense(128, activation='relu')(l1)
proj = layers.Dense(128, activation='linear')(l1)
drop = layers.Dropout(0.4)(l1)
classif = layers.Dense(5, activation='softmax')(drop)
#lout = layers.Dense(5, activation='sigmoid')(l1)
small_classifier = tf.keras.Model(inp, [proj, classif])
small_classifier.load_weights("small_classifier.weights.h5")



generator = RaggedGenerator(4)
#generator.images = generator.images[:20]
#generator.labels = generator.labels[:20]

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
mlp = create_mlp()
adv = create_adversarial_mlp()

optimizer1 = keras.optimizers.Adam(1e-4)
optimizer2 = keras.optimizers.Adam(1e-4)

backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_deepcluster_iter205.weights.h5")
mlp.load_weights("/home/barrage/grp3/models/alvin/mlp_deepcluster_iter205.weights.h5")
#backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_600_adv_50.weights.h5")
#mlp.load_weights("/home/barrage/grp3/models/alvin/mlp_600_adv_50.weights.h5")


print(mlp(np.random.random((1, 2048))))


model = tf.keras.Sequential([
    backbone,
    mlp
])

for layer in model.layers:
    print(layer.name)

print(model.layers[0], model.layers[0].input, model.layers[0].get_layer("conv5_block3_3_conv").output, model.layers[-1].output)
print(model.layers[0].get_layer("conv5_block3_3_conv").name, model.layers[0].get_layer("conv5_block3_3_conv").trainable)


def grad_cam(model, image, class_index, layer_name="conv5_block3_3_conv"):
   
    backbone = model.layers[0]
    mlp = model.layers[1]
    backbone = tf.keras.Model(backbone.input, [backbone.get_layer(layer_name).output, backbone.output])


    image = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        #conv_outputs, predictions = grad_model(image)
        #tape.watch(image)
        conv_outputs, features = backbone(image)
        predictions = mlp(features)
        print(predictions)
        loss = tf.cast(predictions[:, class_index], dtype=tf.float32)  # Prédiction de la classe cible
        print(loss)
    grads = tape.gradient(loss, conv_outputs)
    #print(grads)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    print(pooled_grads, conv_outputs)

    conv_outputs = conv_outputs[0] 
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)  # Relu
    heatmap /= (np.max(heatmap) + 1e-6)
    print(heatmap)
    return heatmap

image_index = 150  #
pred = model.predict(np.expand_dims( (generator.test_images[image_index]/255.0).astype(np.float16), axis=0))
print(pred)
class_index = np.argmax(pred, axis=1)[0]
print(class_index)

heatmap = grad_cam(model, generator.test_images[image_index], class_index)
normalized_heatmap = (heatmap * 255).astype(np.uint8)
print(np.max(normalized_heatmap), np.min(normalized_heatmap))
heatmap = Image.fromarray(normalized_heatmap)  # Conversion en image PIL
heatmap = heatmap.resize((600, 600), Image.LANCZOS)  # Redimensionnement


heatmap = np.array(heatmap).astype(np.uint8)  # Reconvertir en NumPy
colormap = plt.get_cmap("jet")  # Utilisation de Matplotlib pour la colormap
heatmap = colormap(heatmap / 255.0)  # Normalisation entre 0 et 1
heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)  # Garder RGB, enlever alpha
original_img = Image.fromarray(generator.test_images[image_index].astype(np.uint8))  # Image originale
heatmap_img = Image.fromarray(heatmap)  # Heatmap convertie en image PIL
superimposed_img = Image.blend(original_img.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=0.4)


# ---- Afficher ----
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(generator.test_images[image_index].astype(np.uint8))
plt.axis("off")
plt.title("Image originale")

plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.axis("off")
plt.title(f"Grad-CAM (Classe {class_index})")

plt.savefig("gradcam"+str(image_index)+".png")


import numpy as np
import matplotlib.pyplot as plt

def extract_high_activation_regions(image, heatmap, threshold=0.8, margin=200, min_iou=0.3):
    binary_map = (heatmap > threshold).astype(np.uint8)
    labeled_array, num_features = scipy.ndimage.label(binary_map)
    slices = scipy.ndimage.find_objects(labeled_array)
    if num_features == 0:
        print("⚠️ Aucune activation forte détectée, retourne l'image entière")
        return [image] 
    boxes = []
    for s in slices:
        y_start, x_start = s[0].start, s[1].start
        y_end, x_end = s[0].stop, s[1].stop
        x_start = max(x_start - margin, 0)
        y_start = max(y_start - margin, 0)
        x_end = min(x_end + margin, image.shape[1])
        y_end = min(y_end + margin, image.shape[0])

        boxes.append((x_start, y_start, x_end, y_end))

    filtered_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        keep = True
        for fbox in filtered_boxes:
            fx1, fy1, fx2, fy2 = fbox
            inter_x1 = max(x1, fx1)
            inter_y1 = max(y1, fy1)
            inter_x2 = min(x2, fx2)
            inter_y2 = min(y2, fy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            box_area = (x2 - x1) * (y2 - y1)
            fbox_area = (fx2 - fx1) * (fy2 - fy1)
            iou = inter_area / float(box_area + fbox_area - inter_area)
            if iou > min_iou:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)

    crops = [image[y1:y2, x1:x2] for (x1, y1, x2, y2) in filtered_boxes]

    return crops

crops = extract_high_activation_regions(generator.test_images[image_index].astype(np.uint8), heatmap)
print(len(crops))
# ---- Afficher toutes les régions activées ----
plt.figure(figsize=(12, 6))
for i, crop in enumerate(crops):  # Limite à 5 crops
    plt.subplot(1, len(crops), i + 1)
    plt.imshow(crop)
    plt.axis("off")
    plt.title(f"Zone {i+1}")

plt.suptitle("Crops des zones fortement activées (filtrées)", fontsize=16)
plt.savefig("gradcrops"+str(image_index)+".png")




#toto()


### 0  : abdomen   1 : antenne  2 : pattes   3 :   tete     4 : background


train_images = (generator.images / 255.0).astype(np.float16)
train_labels = generator.labels

id_to_member = {0:"abdomen", 1:"antenne", 2:"pattes", 3:"tete", 4:"background"}



ref_activation = {}
for i in range(9) :
    ref_activation[i] = {"abdomen":0, "antenne":0, "pattes":0, "tete":0, "background":0}

for im in train_images :

    
    class_index = np.argmax(model.predict(np.expand_dims(im, axis=0)))   # on prédit la classe de train

    heatmap = grad_cam(model, im, class_index)   # 

    crops = extract_high_activation_regions((im * 255).astype(np.uint8), heatmap)
    print(len(crops), crops)
    
    if len(crops) > 0 :

        crops = np.array([np.array(Image.fromarray(crop).resize((200, 200), Image.BILINEAR)) for crop in crops]) /255.0
        print(crops.shape)
        proj, parties_prob = small_classifier(crops, training=False)
        print(parties_prob)
        parties = np.argmax(parties_prob, axis=1)
        for part in parties :
            ref_activation[class_index][id_to_member[part]] += 1

with open('ref_activation.json', 'w') as f:
    json.dump(ref_activation, f)

# Charger le dictionnaire à partir du fichier JSON
with open('ref_activation.json', 'r') as f:
    loaded_ref_activation = json.load(f)

print(loaded_ref_activation)


































