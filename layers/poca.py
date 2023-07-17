import tensorflow as tf
keras = tf.keras

from keras.layers import *
from keras.models import *


def poca(x, p, ver_x, ver_p):
    v = tf.linalg.cross(p, ver_p)

    m = tf.stack([p, -ver_p, v], axis=-1)
    b = ver_x - x

    m_inv = tf.linalg.pinv(m)
    t = tf.linalg.matmul(m_inv, b[..., tf.newaxis])
    scattered = (tf.linalg.det(m) > 1e-8) | (tf.linalg.det(m) < -1e-8)
    not_scattered = tf.cast(~scattered, tf.float32)[..., tf.newaxis]
    scattered = tf.cast(scattered, tf.float32)[..., tf.newaxis]
    t, ver_t, _ = tf.unstack(tf.squeeze(t, axis=-1), axis=-1)
    t = t[..., tf.newaxis]
    ver_t = ver_t[..., tf.newaxis]

    poca_points = (x + p * t + ver_x + ver_t * ver_p) / 2
    poca_points = poca_points * scattered + (x + ver_x) / 2 * not_scattered

    return poca_points


# a neural network version of poca :)
def poca_nn(layers=3, d=64, k=5, name="poca_nn"):
    muons = Input((None, 12,))
    dosage = tf.shape(muons)[1]

    x = muons

    for i in range(layers):
        x = Dense(2*d, activation="swish", name=f"{name}_hidden_layer_{i}")(x)
        x = Dense(d, activation="swish", name=f"{name}_projection_{i}")(x)
        feature_vector = tf.math.reduce_mean(x, axis=1, keepdims=True)

        x = tf.concat([muons, tf.repeat(feature_vector, dosage, axis=1)], axis=-1)

    outputs = Dense(1 + 3*k)(x)

    return Model(inputs=muons, outputs=outputs)


if __name__ == "__main__":
    model = poca_nn()
