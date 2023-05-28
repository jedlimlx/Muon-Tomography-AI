import tensorflow as tf
keras = tf.keras

from keras.layers import *


def Triplane(filters, kernel_size, padding="same", groups=1, name=None, activation=None):
    """
    Special type of convolution for 3D images (2D convolutions in 3 orientations)
    """

    if name is None: name = "triplane"

    def apply(x):
        def f(x, i):
            x = tf.reshape(x, (-1, shape[1], shape[2], shape[3] * shape[4]))
            x = Conv2D(filters, kernel_size, padding=padding, groups=groups, name=f"{name}_conv2d_{i}")(x)
            x = tf.reshape(x, (-1, shape[1], shape[2], shape[3], shape[4]))
            return x

        shape = x.shape

        # (b, x, y, z, c) -> (b, x, y, z * c)
        x = f(x, 0)

        # (b, x, y, z, c) -> (b, x, z, y, c)
        x = tf.transpose(x, (0, 1, 3, 2, 4))
        x = f(x, 1)
        x = tf.transpose(x, (0, 1, 3, 2, 4))

        # (b, x, y, z, c) -> (b, z, y, x, c)
        x = tf.transpose(x, (0, 3, 2, 1, 4))
        x = f(x, 2)
        x = tf.transpose(x, (0, 3, 2, 1, 4))

        if activation is not None:
            x = Activation(activation)(x)

        return x
    return apply
