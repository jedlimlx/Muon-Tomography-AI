import numpy as np
import tensorflow as tf
import skimage


def random_shapes(ds_size, h, w):
    MU_WATER = 20
    MU_AIR = 0.02
    MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
    for i in range(ds_size):
        img = skimage.draw.random_shapes(image_shape=(h, w), max_shapes=10, min_shapes=1, min_size=w//5, max_size=w//2,
                                         intensity_range=(20, 200), channel_axis=None, allow_overlap=True)[0]
        yield tf.convert_to_tensor(img, dtype=tf.float32) / 255 * MU_MAX


def test():
    import matplotlib.pyplot as plt
    MU_WATER = 20
    MU_AIR = 0.02
    MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
    for i in random_shapes(8, 256, 256):
        plt.imshow(i.numpy() / MU_MAX)
        plt.show()
