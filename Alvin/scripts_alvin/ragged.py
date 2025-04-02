import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras import layers
import os
from PIL import Image
import random
n = '4'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3})


def create_model() :
    inp = keras.Input((None, None, 3))
    c1 = layers.Conv2D(64, activation='relu', kernel_size=3)(inp)
    c1 = layers.Conv2D(64, activation='relu', kernel_size=3)(c1)
    c1 = layers.Conv2D(128, activation='relu', kernel_size=3)(c1)
    c1 = layers.Conv2D(128, activation='relu', kernel_size=3)(c1)
    c1 = layers.Conv2D(256, activation='relu', kernel_size=3)(c1)
    c1 = layers.Conv2D(256, activation='relu', kernel_size=3)(c1)
    c1 = layers.Conv2D(512, activation='relu', kernel_size=3)(c1)
    c = layers.GlobalAveragePooling2D()(c1)
    l = layers.Dense(256, activation='relu')(c)
    l = layers.Dense(256, activation='relu')(l)
    l = layers.Dense(9, activation='softmax')(l)
    return keras.Model(inp, l)



class RaggedGenerator(keras.utils.Sequence) :
    def __init__(self, batch_size=32) :
        self.batch_size=batch_size
        self.max_background = 300
        self.backgrounds = []
        self.images = []
        self.labels = []
        self._load_background("/home/barrage/grp3/crops/background_patches/")
        print("background loaded")
        self._load_from_csv("/home/barrage/grp3/crops/raw_crops/", "labels.csv")
        print("images loaded")
        self._convert_images()

        self.on_epoch_end()


    def _load_background(self, folder):
        all_imgs = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        selected_imgs = random.sample(all_imgs, min(self.max_background, len(all_imgs)))

        for img_name in selected_imgs:
            img_path = os.path.join(folder, img_name)
            labels = np.zeros((9))
            labels[8] = 1
            img = Image.open(img_path).convert("RGB")
            img = np.array(img) / 255.0 
            self.background.append(img)
            sef.images.append(img)
            self.labels.append(labels)


    def _load_from_csv(self, data_dir, csv_path):
        df = pd.read_csv(data_dir+csv_path)
        max_h = 0
        max_w = 0
        for _, row in df.iterrows():
            labels = [int(row["label1"]), int(row["label2"]), int(row["label3"]), int(row["label4"])]
            img_name = row["img_name"]
            img_path = os.path.join(data_dir, img_name)
            label = np.zeros((9))
            for idx in labels :
                label[idx] = 1
            label /= np.sum(label) 
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img = np.array(img)/255.0
                h, w = img.shape[:2]
                max_h = max(h, max_h)
                max_w = max(w, max_w)
                self.images.append(img)
                self.labels.append(label)
        self.max_h = max_h + 10
        self.max_w = max_w + 10



    def augment_image(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        return image


    def __len__(self) :
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        batch_img = self.images[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size : (idx+1)*self.batch_size]

        for i in range(len(batch_img)) :
            im = batch_img[i]
            bg = self.background[np.random.randint(len(self.background))]
            bg = Image.fromarray(bg)
            bg = bg.resize((self.max_w, self.max_h), Image.BILINEAR)
            w, h = im.size

        # Calcul des offsets pour centrer l'image sur le background
            x_offset = (self.max_w - w) // 2
            y_offset = (self.max_h - h) // 2

        # Coller l'image sur le background
            bg.paste(im, (x_offset, y_offset))

        # Redimensionner en 224x224
            final_img = bg.resize((224, 224), Image.BILINEAR)

            batch_img[i] = final_img

        return self.augment_image(np.array(batch_img)), tf.convert_to_tensor(batch_labels)


    def on_epoch_end(self):
        indices = np.arange(len(self.images))
        random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]









data_gen = RaggedGenerator()
print(len(data_gen.images))
optimizer = keras.optimizers.Adam(1e-4)
model = create_model()
model.compile(optimizer=optimizer, loss="categorical_crossentropy")
model.fit(data_gen, epochs=10)
model.save("model1.h5")



toto()





batch_size = 32
for epoch  in range(10) :
  count = 0
  while count < len(data_gen) :

    with tf.GradientTape() as tape :
      gt = []
      predictions = []
      for i in range(batch_size) :
        x, y = data_gen[count]
        pred = model(x, training=True) 
        predictions.append(pred)
        gt.append(y)
        count+=1
      predictions = tf.concat(predictions, axis=0)
      gt = tf.concat(gt, axis=0) 
      loss = tf.reduce_mean(keras.losses.categorical_crossentropy(gt, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print(loss)


model.save("model1.h5")
