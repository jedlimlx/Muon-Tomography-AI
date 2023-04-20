import tensorflow as tf
keras = tf.keras

from keras.layers import *

import numpy as np
import tensorflow_addons as tfa


def skip_connection_2d_to_3d(filters, activation="relu", name=None):
    """
    The skip connections in the U-Net
    Args:
        filters: Number of output filters
        activation: default 'relu', activation to use for Conv layers
        name: string, name of block

    Returns: Function that applies the skip connection layer.

    """
    def apply(x):
        x = Conv2D(x.shape[2], 3, activation=activation, padding="same", name=name + "_conv2d")(x)
        x = Reshape((x.shape[1], x.shape[2], x.shape[2], 1), name=name + "_reshape")(x)
        x = Conv3D(filters, 3, activation=activation, padding="same", name=name + "_conv3d")(x)

        return x

    return apply


class Cart2Polar(Layer):
    def __init__(self, angle_res=1024, r_res=256, name='cart2polar', **kwargs):
        super().__init__(name=name, **kwargs)

        self.angle_res = angle_res
        self.r_res = r_res
        self.coords = None

    def build(self, input_shape):
        n, h, w, c = input_shape
        origin = tf.convert_to_tensor((w / 2, h / 2), dtype=tf.float32)

        angles = tf.expand_dims(tf.linspace(-np.pi / 2, np.pi / 2, self.angle_res), -1)
        r = tf.expand_dims(tf.linspace(-1., 1., self.r_res), 0)
        r = tf.abs(r) * tf.sqrt(float(h * h + w * w)) / 2 * tf.sign(r)

        x = r * tf.cos(angles)
        y = r * tf.sin(angles)

        coords = tf.stack([x, y], axis=-1) + origin
        self.coords = tf.reshape(coords, (1, -1, 2))

    def call(self, inputs, *args, **kwargs):
        return tf.reshape(tfa.image.interpolate_bilinear(inputs, self.coords, indexing='xy'),
                          (-1, self.angle_res, self.r_res, 1))

    def get_config(self):
        cfg = super(Cart2Polar, self).get_config()
        cfg.update({
            'angle_res': self.angle_res,
            'r_res': self.r_res
        })
        return cfg


def cart2polar(img, angle_res=1024, r_res=256, angle_range='180'):
    assert angle_range in ['180', '360']
    n, h, w, c = img.shape

    # xy indexing because ij gets confusing very fast
    origin = tf.convert_to_tensor((w / 2, h / 2), dtype=tf.float32)
    if angle_range == '180':
        angles = tf.expand_dims(tf.linspace(-np.pi / 2, np.pi / 2, angle_res), -1)
        r = tf.expand_dims(tf.linspace(-1., 1., r_res), 0)
        r = tf.abs(r) * tf.sqrt(float(h * h + w * w)) / 2 * tf.sign(r)
    else:
        angles = tf.expand_dims(tf.linspace(0., 2 * np.pi, angle_res), -1)
        r = tf.expand_dims(tf.linspace(0., 1., r_res), 0)
        r = tf.sqrt(r) * tf.sqrt(float(h * h + w * w)) / 2

    x = r * tf.cos(angles)
    y = r * tf.sin(angles)

    coords = tf.stack([x, y], axis=-1) + origin
    coords = tf.reshape(coords, (1, -1, 2))

    return tf.reshape(tfa.image.interpolate_bilinear(img, coords, indexing='xy'), (-1, angle_res, r_res, 1))


class Polar2Cart(Layer):
    def __init__(self, h=362, w=362, name='polar2cart', **kwargs):
        super().__init__(name=name, **kwargs)

        self.h = h
        self.w = w

        self.coords = None

    def build(self, input_shape):
        n, angle_res, r_res, c = input_shape

        x, y = tf.meshgrid(
            tf.range(0, self.w, dtype=tf.float32),
            tf.range(0, self.h, dtype=tf.float32)
        )
        origin = tf.convert_to_tensor((self.w / 2, self.h / 2), dtype=tf.float32)
        x = x - origin[0]
        y = y - origin[1]

        r = tf.sqrt(x * x + y * y)
        r = r / tf.sqrt(float(self.h * self.h + self.w * self.w))
        th = tf.math.atan2(y, x)

        r = r * tf.sign(tf.cos(th)) + 0.5
        r = r * (r_res - 1)
        th = tf.math.atan(tf.math.tan(th)) + np.pi / 2
        th = th * (angle_res - 1) / np.pi

        coords = tf.stack([th, r], axis=-1)
        self.coords = tf.reshape(coords, (1, -1, 2))

    def call(self, inputs, *args, **kwargs):
        return tf.reshape(tfa.image.interpolate_bilinear(inputs, self.coords, indexing='ij'),
                          (-1, self.h, self.w, 1))

    def get_config(self):
        cfg = super(Polar2Cart, self).get_config()
        cfg.update({
            'h': self.h,
            'w': self.w
        })
        return cfg


def polar2cart(img, h=512, w=512, angle_range='180'):
    assert angle_range in ['180', '360']
    n, angles_res, r_res, c = img.shape

    # xy indexing because ij gets confusing very fast
    x, y = tf.meshgrid(
        tf.range(0, w, dtype=tf.float32),
        tf.range(0, h, dtype=tf.float32)
    )
    origin = tf.convert_to_tensor((w / 2, h / 2), dtype=tf.float32)
    x = x - origin[0]
    y = y - origin[1]

    if angle_range == '180':
        r = tf.sqrt(x * x + y * y)
        r = r / tf.sqrt(float(h * h + w * w))
        th = tf.math.atan2(y, x)

        r = r * tf.sign(tf.cos(th)) + 0.5
        r = r * (r_res - 1)
        th = tf.math.atan(tf.math.tan(th)) + np.pi / 2
        th = th * (angles_res - 1) / np.pi
    else:
        r = x * x + y * y
        r = r / (h * h + w * w) * 4 * r_res
        th = tf.math.atan2(y, x) + np.pi
        th = th * angles_res / np.pi

    print(tf.reduce_sum(tf.cast(tf.math.is_inf(th), tf.int32)))
    coords = tf.stack([th, r], axis=-1)
    coords = tf.reshape(coords, (1, -1, 2))

    return tf.reshape(tfa.image.interpolate_bilinear(img, coords, indexing='ij'), (-1, h, w, 1))


def main():
    import matplotlib.pyplot as plt
    from data_generation import random_shapes
    img = tf.reshape(next(random_shapes(1, 362, 362)), (1, 362, 362, 1))
    plt.imshow(img[0], cmap='gray')
    plt.show()

    polar = Cart2Polar(angle_res=1024, r_res=512)(img)
    print(polar[:, 0, 0, :])
    plt.imshow(polar[0], cmap='gray')
    plt.show()

    # print(tf.reduce_sum(tf.cast(tf.math.is_nan(polar2cart(polar, h=256, w=256)), tf.int32)))
    plt.imshow(Polar2Cart(h=362, w=362)(polar)[0], cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
