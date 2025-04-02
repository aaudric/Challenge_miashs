import tensorflow as tf
import tensorflow.keras as keras    
from tensorflow.keras import layers 
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def create_model() :
    inp = keras.Input((None, None, 3))
    c1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(inp)
    c1 = layers.LeakyReLU()(c1)
    c1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(c1)
    c1 = layers.LeakyReLU()(c1)
    c1 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(c1)
    c1 = layers.LeakyReLU()(c1)
    c1 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(c1)
    c1 = layers.LeakyReLU()(c1)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1)
    c1 = layers.LeakyReLU()(c1)
    c1 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(c1)
    c1 = layers.LeakyReLU()(c1)
    c1 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same')(c1)
    c1 = layers.LeakyReLU()(c1)
    c1 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same')(c1)
    c1 = layers.LeakyReLU()(c1)
    c1 = layers.Conv2D(1024, kernel_size=3, strides=1, padding='same')(c1)
    c1 = layers.LeakyReLU()(c1)
    c = layers.GlobalAveragePooling2D()(c1)
    l1 = layers.Dense(512)(c)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dense(512)(c)
    l1 = layers.LeakyReLU()(l1)
    out = layers.Dense(9, activation='softmax')(l1)
    return keras.Model(inp, out)

model = create_model()



model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(1e-4))
model.fit(, epochs=10, batch_size=32)