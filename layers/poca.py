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


def loss(y, y_pred):
    n = y[:, :, 0:1]
    mask = tf.repeat(
        tf.repeat(
            tf.range(16, dtype=tf.float32)[tf.newaxis, :], tf.shape(y)[1], axis=0
        )[tf.newaxis, ...], tf.shape(y)[0], axis=0
    ) <= 3 * tf.repeat(n, 16, axis=-1)
    return tf.math.reduce_sum(tf.square(y - y_pred) * tf.cast(mask, tf.float32)) / \
        tf.math.reduce_sum(tf.cast(mask, tf.float32))


if __name__ == "__main__":
    # Create a description of the features.
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
        'voxels': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_example(example_proto):
        res = tf.io.parse_single_example(example_proto, feature_description)
        x = tf.io.parse_tensor(res['x'], out_type=tf.double)
        y = tf.io.parse_tensor(res['y'], out_type=tf.double)
        voxels = tf.io.parse_tensor(res['voxels'], out_type=tf.int32)

        x.set_shape((None, 12))
        y.set_shape((None, 1+5*3))
        voxels.set_shape((64, 64, 64))

        return tf.cast(x[:1000], tf.float32), tf.cast(y[:1000], tf.float32)

    ds = tf.data.TFRecordDataset("../scattering_prediction.tfrecord").map(_parse_example).batch(16)
    val_ds = ds.take(10)
    train_ds = ds.skip(10)

    # building model
    model = poca_nn(d=128)
    model.compile(optimizer="adam", loss=loss)

    model.fit(train_ds, epochs=30, validation_data=val_ds)
