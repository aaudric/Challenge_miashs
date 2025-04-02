import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers




class Block(tf.keras.Model) :
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0) :
        super().__init__()
        self.embed_dim=embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)


        self.v = layers.Dense(self.embed_dim, activation='linear', use_bias=False)
        self.q = layers.Dense(self.embed_dim, activation='linear', use_bias=False)
        self.k = layers.Dense(self.embed_dim, activation='linear', use_bias=False)
        self.proj = layers.Dense(self.embed_dim, activation='linear')

        self.mlp1 = layers.Dense(int(self.embed_dim*self.mlp_ratio), activation='gelu')
        self.mlp2 = layers.Dense(self.embed_dim, activation='linear')

    def call(self, x) :
        # layer norm 1
        x_nrom = self.norm1(x)

        ##### ATTENTION PART #####
        b, n, c = tf.shape(x_nrom)[0], tf.shape(x_nrom)[1], tf.shape(x_nrom)[2]
        head_dim = self.embed_dim // self.num_heads
        q = tf.transpose(tf.reshape(self.q(x_nrom) * head_dim**-.5, (b, n, self.num_heads, c // self.num_heads)), perm=[0, 2, 1, 3])
        k = tf.transpose(tf.reshape(self.k(x_nrom), (b, n, self.num_heads, c // self.num_heads)), perm=[0, 2, 1, 3])
        v = tf.transpose(tf.reshape(self.v(x_nrom), (b, n, self.num_heads, c // self.num_heads)), perm=[0, 2, 1, 3])
       

        attn = tf.matmul(q, k, transpose_b=True)  # 1, 8, 65, 65
        attn = tf.nn.softmax(attn, axis=-1)
        y = tf.matmul(attn, v)  # 1, 8, 65, 72
        y = tf.transpose(y, perm=[0, 2, 1, 3])   # 1, 65, 8, 72
        y = tf.reshape(y, (b, n, c))  # shape 1, 65, 576
        y = self.proj(y)  # 1, 65, 576
        x = x + y     # 1, 65, 576      y = résultat du x_norm dans le bloc attention, x pas layer normalisé


        x_norm = self.norm2(x) 
        ##### MLP PART #####   -> pas de drop car régression?
        y = self.mlp1(x_norm)   # 1, 65, 576*mlp_ratio
        y = self.mlp2(y)     # 1, 65, 576
        x = x + y   # 1, 65, 576       y = résultat du x_norm dans bloc attention,     x pas layer normalisé
        
        return x


class PatchExtractor(tf.keras.layers.Layer) :
    def __init__(self, patch_size=4, embed_dim=1024, image_size=64) :
        super().__init__()
        self.patch_size=patch_size
        self.embed_dim = embed_dim
        self.patch_conv = tf.keras.layers.Conv2D(filters=self.embed_dim, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size), padding='valid', activation='linear')
        self.num_patches = (image_size // patch_size) ** 2 +1  # pour cls
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches, output_dim=embed_dim
        )
        self.cls_token = self.add_weight(
            shape=(1, 1, embed_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token"
        )

    def call(self, inputs) :
        res_conv = self.patch_conv(inputs)  
        patch_embedding = tf.reshape(res_conv, (tf.shape(res_conv)[0], -1, tf.shape(res_conv)[-1]))  # shape BATCH, N_PATCH, EMBED_DIM

        batch_size = tf.shape(patch_embedding)[0]
        cls_token = tf.tile(self.cls_token, [batch_size, 1, 1])
        patch_embedding = tf.concat([cls_token, patch_embedding], axis=1)

        positions = tf.expand_dims(
            tf.range(start=0, limit=self.num_patches, delta=1), axis=0
        )
        patch_embedding += self.position_embedding(positions)
        return patch_embedding
    

class ViT_backbone(tf.keras.Model) :
    def __init__(self, embed_dim=1024, num_blocks=4, num_heads=8, patch_size=4, gp='none', mlp_ratio=4.0) :
        super().__init__()
        self.embed_dim=embed_dim
        self.patch_master = PatchExtractor(patch_size, embed_dim=self.embed_dim, image_size=64)
        self.blocks = [Block(embed_dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for i in range(num_blocks)]
        if gp == "average" :
            self.last_pool = layers.GlobalAveragePooling1D()
            self.gp=True
        elif gp == "max" :
            self.last_pool = layers.GlobalMaxPooling1D()
            self.gp=True
        elif gp=="none" :
            self.gp=False

    def call(self, inputs) :
        x = self.patch_master(inputs) 
        for i in range(len(self.blocks)) :
            x = self.blocks[i](x)

        if self.gp :
            return self.last_pool(x)
        else :
            return x[:, 0]   # sortie B, N   => cls token




class ViTGenerator(keras.utils.Sequence) :
    def __init__(self, batch_size=32) :
        self.batch_size=batch_size
        self.max_background = 300
        self.backgrounds = []
        self.images = []
        self.ratios = []
        self.labels = []
       
        self._load_from_csv("/home/barrage/grp3/crops/raw_crops/", "labels.csv")
        self.load_test_images("/home/barrage/grp3/datatest/")
        print("images loaded")
        self.images = np.array(self.images)
        self.ratios = np.array(self.ratios)
        self.labels = np.array(self.labels)
        self.on_epoch_end()


    def load_test_images(self, folder_path):
   
        names = []
        images_resized = []
        original_dimensions = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(".jpg"):
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path)

                img_resized = img.resize((512, 512))
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
        for _, row in df.iterrows():
            labels = [int(row["label1"]), int(row["label2"]), int(row["label3"]), int(row["label4"])]
            img_name = row["img_name"]
            img_path = os.path.join(data_dir, img_name)
            label = np.zeros((9))
            for idx in labels :
                label[idx] = 1
            label /= np.sum(label) 
            if os.path.exists(img_path):
                im = Image.open(img_path).convert("RGB")
                im_np = np.array(im)
                h, w = im_np.shape[:2]

                im = im.resize((512, 512))
                im_np = np.array(im, dtype=np.uint8)
                
                self.images.append(im_np)
                self.ratios.append([h/3000, w/3000])
                la = np.zeros((9))
                la[np.argmax(label)] = 1
                self.labels.append(la)
      
    def __len__(self) :
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        start = idx*self.batch_size 
        stop = (idx+1) * self.batch_size
        batch_img = self.images[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_img = tf.image.random_flip_left_right(batch_img)
        batch_img = tf.image.random_flip_up_down(batch_img)
        batch_img = tf.image.random_brightness(batch_img, max_delta=0.1)
        batch_ratios = self.ratios[idx*self.batch_size : (idx+1)*self.batch_size]   + np.random.randint(-10, 10, (self.batch_size, 2))/3000
        batch_labels = self.labels[idx*self.batch_size : (idx+1)*self.batch_size]


        batch_img_inds = np.arange(len(self.test_images))
        random.shuffle(batch_img_inds)
        batch_img_inds = batch_img_inds[:self.batch_size]
        test_img = self.test_images[batch_img_inds]

        adversarial_labels = tf.expand_dims(tf.concat([tf.ones(self.batch_size), tf.zeros(self.batch_size)], axis=0), axis=1)



        return {"train_img":tf.cast(batch_img, dtype=tf.float16)/255.0, "train_ratio":batch_ratios, "train_labels":batch_labels, "test_img":tf.cast(test_img, dtype=tf.float16)/255.0, "adv_labels":adversarial_labels}


    def on_epoch_end(self):
        indices = np.arange(len(self.images))
        random.shuffle(indices)
        self.images = self.images[indices]
        self.ratios = self.ratios[indices]
        self.labels = self.labels[indices]