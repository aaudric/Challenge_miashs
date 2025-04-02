import tensorflow as tf
import numpy as np
import random
import tensorflow.keras as keras

class Generator(keras.utils.Sequence) :
    def __init__(self, path="../crops/croped_data.npz", batch_size=32) :
        self.batch_size = batch_size
        data = np.load(path, allow_pickle=True)
        self.images = data["images"] / 255.0
        labels = data["labels"]
        labels = labels.item()
        self.labels = []
        for i in range(self.images.shape[0]) :
            counter = np.zeros((9))
            counter[int(labels["label1"][i])] +=1
            counter[int(labels["label2"][i])] +=1
            counter[int(labels["label3"][i])] +=1
            counter[int(labels["label4"][i])] +=1

            classe = np.argmax(counter)
            self.labels.append(classe)
        self.labels = np.array(self.labels)

        indices = np.arange(len(self.images))
        random.shuffle(indices)
        cut = int(0.8*self.images.shape[0])
        train_indices = indices[:cut]
        validation_indices = indices[cut:]
        self.validation_images = self.images[validation_indices]
        self.validation_labels = self.labels[validation_indices]
        self.images = self.images[train_indices]
        self.labels = self.labels[train_indices]

    def __len__(self) :
        return int(self.images.shape[0] // self.batch_size)


    def on_epoch_end(self) :
        indices = np.arange(len(self.images))
        random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]


    def __getitem__(self, idx) :
        batch_img = self.images[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_img = tf.image.random_flip_left_right(batch_img)
        batch_img = tf.image.random_flip_up_down(batch_img)
        batch_img = tf.image.random_brightness(batch_img, max_delta=0.1)  
        batch_img = tf.image.random_contrast(batch_img, lower=0.9, upper=1.1) 

        return tf.convert_to_tensor(batch_img), tf.convert_to_tensor(batch_labels)
        




import pandas as pd
from PIL import Image
class GeneratorCSV(keras.utils.Sequence) :
    def __init__(self, path="/home/barrage/grp3/crops/raw_crops/train_labels.csv", batch_size=1) :
        data = pd.read_csv(path)
        self.batch_size = batch_size

        self.images = []
        self.labels = []
        for i in range(len(data)) :
            labels = np.zeros((9))
            labels[int(data["label1"][i])] = 1
            labels[int(data["label2"][i])] = 1
            labels[int(data["label3"][i])] = 1
            labels[int(data["label4"][i])] = 1

            image = Image.open("/home/barrage/grp3/crops/raw_crops/"+data["img_name"][i])
            image_np = np.array(image)
            self.images.append(image_np)
            self.labels.append(labels)


    def __len__(self) :
        return len(self.images)


    def on_epoch_end(self) :
        indices = np.arange(len(self.images))
        random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]


    def __getitem__(self, idx) :
        batch_img = np.expand_dims(self.images[idx], axis=0)  # 1, N, M, 3
        batch_labels = np.expand_dims(self.labels[idx], axis=0)  # 1, 9

        batch_img = tf.image.random_flip_left_right(batch_img)
        batch_img = tf.image.random_flip_up_down(batch_img)

        return tf.convert_to_tensor(batch_img), tf.convert_to_tensor(batch_labels)



    


class TestGenerator(keras.utils.Sequence) :
    def __init__(self, path, batch_size=128) :
        self.batch_size = batch_size
        