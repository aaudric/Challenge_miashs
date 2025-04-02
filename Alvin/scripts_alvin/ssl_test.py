import tensorflow as tf 
import numpy as np
import tensorflow.keras as keras
import os
from tensorflow.keras import layers
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from PIL import Image
from tensorflow.keras.applications import ResNet50


class SSLGenerator(keras.utils.Sequence) :
    def __init__(self, test_dir="/home/barrage/grp3/datatest/", batch_size=4) :
        self.batch_size = batch_size
        images = []
        max_w = 0
        max_h = 0
        img_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:30]
        back = [f for f in os.listdir()]
        for img in img_files :
            image = Image.open(test_dir+img)
            image_np = np.array(image)
            images.append(image_np)
            max_w = max(image_np.shape[1], max_w)
            max_h = max(image_np.shape[0], max_h)
        max_w += 10
        max_h+=10
        self.images = np.zeros((len(images), 224, 224, 3))
        for j, im in enumerate(images) :
            print(j)
            template = (np.random.random((max_h, max_w, 3))*255).astype(np.uint8)
            h, w = im.shape[:2]
            ymin = int(max_h //2 - h //2)
            ymax = int(max_h //2 + h //2)
            xmin = int(max_w //2 - w //2)
            xmax = int(max_w //2 + w //2)

            if h % 2 == 1 :
                ymax+=1
            if w % 2 == 1 :
                xmax += 1
            template[ymin:ymax, xmin:xmax]  = images[j]
            #print(template)
            template_pil = Image.fromarray((template).astype(np.uint8))  # Conversion NumPy -> PIL
            template_resized = np.array(template_pil.resize((224, 224), Image.BILINEAR)) / 255.0
            self.images[j] = template_resized

    def __len__(self) :
        return self.images.shape[0] // self.batch_size


    def __getitem__(self, idx) :
        batch = self.images[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = tf.tile(batch, (2, 1, 1, 1))
        batch = tf.image.random_flip_up_down(batch)
        batch = tf.image.random_flip_right_left(batch)
        batch = batch + tf.random.normal((256, 128, 128, 3), mean=0.0, stddev=0.1)

        batch = tf.image.random_brightness(batch, max_delta=0.2)  # max_delta ∈ [0, 1]
        batch = tf.image.random_contrast(batch, lower=0.8, upper=1.2)

        # Rotation aléatoire (0°, 90°, 180° ou 270°)
        # Rotation image par image
        batch_rotated = []
        for i in range(batch.shape[0]):
            k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            batch_rotated.append(tf.image.rot90(batch[i], k=k))

        batch = tf.stack(batch_rotated)

        # Zoom aléatoire (crop + resize)
        crop_size = tf.random.uniform([], minval=96, maxval=128, dtype=tf.int32)  # Crop entre 96x96 et 128x128
        batch = tf.image.resize_with_crop_or_pad(batch, crop_size, crop_size)
        batch = tf.image.resize(batch, (128, 128))  # Resize en 128x128

        return batch


def projection_mlp() :
    inp = keras.Input((2048,))
    l1 = layers.Dense(512)(inp)
    l1 = layers.LeakyReLU()(l1)
    l1 = layers.Dense(512)(l1)
    l1 = layers.LeakyReLU()(l1)
    out = layers.Dense(256, activation='linear')(l1)
    return keras.Model(inp, out)



class BarlowTwins(keras.Model) :
    def __init__(self, backbone, head, lam=5e-3) : 
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.lam = lam
        self.bn = layers.BatchNormalization()
    
    def call(self, inputs, training=True) :
        x = self.backbone(inputs, training=training)
        z = self.head(x, training=training)
        return self.bn(z, training=training)

    def train_step(self, data) :
        images = data
        with tf.GradientTape() as tape :
            z = self(images)

            loss = self.loss(z, self.lam)

        gradients = tape.gradient(loss, self.backbone.trainable_variables+self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.backbone.trainable_variables+self.head.trainable_variables))

        return {"barlow_twin_loss":loss}

class BarlowTwinsLoss(keras.losses.Loss) :
    def __init__(self) :
        super().__init__()
    
    def call(self, z, lam) :
        z1, z2 = tf.split(z, 2, 0)
        c = tf.matmul(z1, z2, transpose_a=True)
        batch_size = tf.shape(z1)[0]
        c = c / tf.cast(batch_size, c.dtype)
        on_diag = tf.reduce_sum(tf.square(tf.linalg.diag_part(c) - 1))
        off_diag = tf.reduce_sum(tf.square(c - tf.linalg.diag(tf.linalg.diag_part(c))))
        loss = on_diag + lam * off_diag
        return loss         

encoder = ResNet50(input_shape=(128, 128, 3), weights='imagenet', include_top=False, pooling='avg')
projection = projection_mlp()

data_gen = SSLGenerator()

model = BarlowTwins(encoder, projection)
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=BarlowTwinsLoss())


model.fit(data_gen, epochs=10)
























































        

        


