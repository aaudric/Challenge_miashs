import numpy as np
import xgboost as xgb
from tensorflow.keras.applications import ResNet50
import os
import random
from PIL import Image
import pandas as pd
import tensorflow.keras as keras
from sklearn.metrics import f1_score
os.sched_setaffinity(0, {0, 1, 2, 3}) 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class RaggedGenerator(keras.utils.Sequence) :
    def __init__(self, batch_size=32, only_sure=True) :
        self.batch_size=batch_size
        self.max_background = 300
        self.backgrounds = []
        self.images = []
        self.ratios = []
        self.labels = []
        self.only_sure = only_sure
       
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

                img_resized = img.resize((600, 600))
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

            if self.only_sure :

                if np.any(label==4) :

                    if os.path.exists(img_path):
                        im = Image.open(img_path).convert("RGB")
                        im_np = np.array(im)
                        h, w = im_np.shape[:2]

                        im = im.resize((600, 600))
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
            else :
                if os.path.exists(img_path):
                    im = Image.open(img_path).convert("RGB")
                    im_np = np.array(im)
                    h, w = im_np.shape[:2]

                    im = im.resize((600, 600))
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
        self.ratios = self.ratios[indices]
        self.labels = self.labels[indices]


generator = RaggedGenerator()

backbone = keras.applications.ResNet50(input_shape=(600, 600, 3), weights="imagenet", include_top=False, pooling='avg')
backbone.load_weights("/home/barrage/grp3/models/alvin/backbone_600_adv_80.weights.h5")

train_images = generator.images.astype(np.float16) / 255.0
train_features = backbone.predict(train_images)
labels = np.argmax(generator.labels)  ### one hot encodé de base

print("FEATURES COMPUTED")

def f1_eval(preds, dtrain):
    labels = dtrain.get_label()
    preds_labels = np.argmax(preds.reshape(len(labels), -1), axis=1)
    return 'f1', f1_score(labels, preds_labels, average='macro')


xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softprob',  # Utilise softmax pour prédire les probabilités pour chaque classe
    num_class=9,                # Indique le nombre de classes
    use_label_encoder=False,
    eval_metric=f1_eval,  # Tu peux laisser mlogloss ici, mais F1 sera utilisé pour l'évaluation
    n_jobs=4  # Limite l'utilisation des CPU à 4 threads
)

params = {
    "objective": "multi:softprob",
    "num_class": 9,
    "eta": 0.1,
    "max_depth": 6,
    "eval_metric": "mlogloss",  # Utilisé pour l'optimisation
    "n_jobs": 4
}

dtrain = xgb.DMatrix(train_features, label=labels)


xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, "train")],
    feval=f1_eval,  # Suivi de la F1
)

"""
# Entraînement du modèle avec un callback pour le F1-score
xgb_model.fit(
    train_features, 
    labels,
    #eval_metric = f1_eval,
    eval_set=[(train_features, labels)],  # Evaluer sur l'ensemble d'entraînement (ou un ensemble de validation)
)
"""

test_images = generator.test_images.astype(np.float16) / 255.0
test_features = backbone.predict(test_images)
test_features = np.concatenate([test_features, generator.test_dimensions], axis=1)  # B, 2050


predictions = xgb_model.predict(test_features)
predictions = tf.argmax(predictions, axis=1).numpy()

test_names = generator.test_names
    
df = pd.DataFrame({"name": test_names, "label": predictions})
df.to_csv("result_xgboost_base.csv")