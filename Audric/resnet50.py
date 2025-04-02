import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import random

n = '4'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3}) 
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

class PerImage4LabelsBatchGeneratorFromCSV(Sequence):
    def __init__(self, background_dir, data_dir, csv_path, batch_size=4, shuffle=True, max_background=300):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_background = max_background
        self.samples = []

        self._load_background(background_dir)
        self._load_from_csv(data_dir, csv_path)

        if self.shuffle:
            np.random.shuffle(self.samples)

    def _load_background(self, folder):
        all_imgs = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        selected_imgs = random.sample(all_imgs, min(self.max_background, len(all_imgs)))

        for img_name in selected_imgs:
            img_path = os.path.join(folder, img_name)
            txt_path = img_path.replace(".jpg", ".txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path) as f:
                line = f.readline().strip().split()
                label_str = line[0]
                labels = [int(l) for l in label_str.split("_")]
                if len(labels) == 4:
                    self.samples.append((img_path, labels))

    def _load_from_csv(self, data_dir, csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            labels = [int(row["label1"]), int(row["label2"]), int(row["label3"]), int(row["label4"])]
            img_name = row["img_name"]
            img_path = os.path.join(data_dir, img_name)
            if os.path.exists(img_path):
                self.samples.append((img_path, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, labels = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        img_np = np.array(img).astype("float32") / 255.0

        batch_images = np.stack([img_np] * 4, axis=0)
        batch_labels = np.array(labels, dtype=np.int32)

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

def build_resnet50_classifier(num_classes=9):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(None, None, 3)
    )

    base_model.trainable = True  # ou False si tu veux fine-tuner plus tard

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512),
        layers.PReLU(),
        layers.Dense(512),
        layers.PReLU(),
        layers.Dense(num_classes, activation='softmax')  # 9 classes
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model


generator = PerImage4LabelsBatchGeneratorFromCSV(
    background_dir="../../barrage/grp3/crops/background_patches/",
    data_dir="../../barrage/grp3/crops/raw_crops/",
    csv_path="../../barrage/grp3/crops/raw_crops/labels.csv"
)

model = build_resnet50_classifier(num_classes=9)

model.fit(
    generator,
    epochs=10
)

model.save("resnet50_9classes.h5")

