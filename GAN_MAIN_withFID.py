import os
import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from GAN_func import train_step
import numpy as np
import FID

path = r"a_directory_path"

epochs = 10
batch_size = 32
FID_SIZE = 100
fd = FID.init_fid()
FID_list = []

(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))

DataS = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = DataS.shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(batch_size)

for epoch in range(epochs):
    print(r"\n Start epoch", epoch)
    for step, real_images in enumerate(dataset):
        d_loss, g_loss, generated_images = train_step(real_images, batch_size)
        if step % 1000 == 0:
            Img = generated_images[0:9] * 255.0
            for i in range(9):
                plt.subplot(330 + 1 + i)
                plt.imshow(Img[i])
            plt.show(block=False)
            plt.savefig(path + '\Generated_Images\Epoch#' + str(epoch) + 'step#' + str(step) + ".png", bbox_inches='tight')
            plt.close()
            DSet = DataS.batch(200)
            batch_S = 100
            gan_fid = FID.calculate_fid(fd, real_images.numpy(), generated_images.numpy())
            print("Epoch: ", epoch, "Instances:", step, "FID:", gan_fid)
            FID_list.append(gan_fid)