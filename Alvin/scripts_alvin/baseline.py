import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from generator import Generator
from tensorflow.keras.applications import ResNet50
n = '4'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3})



model = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='max')
x = keras.layers.Dense(1024, activation='relu')(model.output)
x = keras.layers.Dense(9, activation='softmax')(x)
model = keras.Model(model.input, x)

data_gen = Generator()


model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

val_acc = tf.keras.metrics.Accuracy()
for i in range(10) :
    model.fit(data_gen, epochs=2)

    validation_predictions = model.predict(data_gen.validation_images)
    val_acc.update_state(data_gen.validation_labels, validation_predictions)
    print("ACCURACY SUR LE JEU DE VALIDATION :", val_acc.result())
    val_acc.reset_state()

model.save("baseline_resnet.h5")


