import random
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


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


class NTXent(tf.keras.losses.Loss) :
    def __init__(self, normalize=True) :
        super().__init__()
        self.large_num = 1e8
        self.normalize = normalize

    def call(self, batch, temperature=1) :
        # sépare les images x des images x'
        hidden1, hidden2 = tf.split(batch, 2, 0)
        batch_size = tf.shape(hidden1)[0]
        if self.normalize :
            hidden1 = tf.nn.l2_normalize(hidden1, axis=1)
            hidden2 = tf.nn.l2_normalize(hidden2, axis=1)
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size*2)    # matrice des labels,    batch_size x 2*batch_size
        masks = tf.one_hot(tf.range(batch_size), batch_size)       # mask de shape     batch x batch

        logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature       ### si normalisé cela aurait été cosine sim => shape batch, batch    distance x x
        logits_aa = logits_aa - masks * self.large_num    ### on rempli la diagonale de très petite valeur car forcément cosine sim entre vecteurs identique = 1
        logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * self.large_num    ###  idem ici ==> donc là on fait distances entre x' x'
        logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature     ### sim x x'
        logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature     ### sim x' x

        loss_a = tf.nn.softmax_cross_entropy_with_logits(              ### matrice labels contient info de où sont les paires positives
            labels, tf.concat([logits_ab, logits_aa], 1))              ### en concaténant ab et aa on obtient similarité de a vers toutes les autres images (en ayant mis sa propre correspondance à 0)
        loss_b = tf.nn.softmax_cross_entropy_with_logits(              ### idem de b vers toutes les images
            labels, tf.concat([logits_ba, logits_bb], 1))
        loss = tf.reduce_mean(loss_a + loss_b)     ### moyenne des 2 et loss

        return loss


class Generator(tf.keras.utils.Sequence) :
  def __init__(self) :
    self.batch_size = 4
    self.paths = ["abdomen1", "abdomen2", "abdomen3", "abdomen4", "antenne1", "antenne2", "antenne3", "antenne4", "pattes1",
                  "pattes2", "pattes3", "pattes4", "tete2", "tete1", "tete3", "tete4", "bg1", "bg2", "bg3", "bg4"]
    self.labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    self.load_images()

  def load_images(self) :
    self.images = []
    for path in self.paths :
      im = Image.open("./"+path+".png").resize((200, 200), Image.BILINEAR)
      self.images.append(np.array(im)/255.0)

    self.images = np.array(self.images)

  def __len__(self) :
    return len(self.images) // self.batch_size


  def __getitem__(self, idx) :
    batch_img = self.images[idx*self.batch_size : (idx+1)*self.batch_size]
    print(batch_img.shape)

    batch_img = tf.concat([batch_img, batch_img], axis=0)
    #batch_img = tf.tile(batch_img, multiples=[2, 1, 1, 1])
    batch_img = tf.image.random_flip_left_right(batch_img)
    batch_img = tf.image.random_flip_up_down(batch_img)
    batch_img = tf.image.random_brightness(batch_img, max_delta=0.1)
    batch_img = tf.image.random_contrast(batch_img, lower=0.6, upper=1.4)
    batch_img = tf.image.random_saturation(batch_img, lower=0.9, upper=1.1)
    batch_labels = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
    batch_labels = tf.concat([batch_labels, batch_labels], axis=0)
    #batch_labes = tf.tile(batch_labels, [2, 1])

    return batch_img, batch_labels

  def on_epoch_end(self):
    inds = np.arange(len(self.images))
    random.shuffle(inds)
    self.images = self.images[inds]
    self.labels = self.labels[inds]


loss_fn = NTXent()
generator = Generator()
print(len(generator))
optimizer = tf.keras.optimizers.Adam(1e-3)

loss_tracker = tf.keras.metrics.Mean()
loss_tracker2 = tf.keras.metrics.Mean()
acc_tracker = tf.keras.metrics.CategoricalAccuracy()

for epoch in range(20) :
  loss_tracker.reset_states()
  loss_tracker2.reset_states()
  acc_tracker.reset_states()
  for i in range(len(generator)) :
    batch = generator[i]
    x, y = batch
    with tf.GradientTape() as tape :
      proj, pred = small_classifier(x, training=True)
      con_loss = loss_fn.call(proj, 0.4)
      classif_loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred) *0.1
      loss = con_loss + classif_loss
    gradients = tape.gradient(loss, small_classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradients, small_classifier.trainable_variables))
    loss_tracker.update_state(con_loss)
    acc_tracker.update_state(y, pred)
    loss_tracker.update_state(classif_loss)
  print("EPOCH ENDED", loss_tracker.result(), loss_tracker2.result(), acc_tracker.result())
  generator.on_epoch_end()



small_classifier.save_weights("small_classifier.weights.h5")