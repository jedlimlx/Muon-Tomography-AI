import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers.agg_2d import MLP


class GaussianScatterAndAvg3D(Layer):
    def __init__(self, resolution, channels, hidden_layers=(8, 4),
                 point_size=5, projection_dim=None, use_lstm=False, std=0.6/64, **kwargs):
        super().__init__(**kwargs)
        self.resolution = resolution
        self.point_size = point_size
        self.projection_dim = projection_dim
        self.channels = channels

        self.std = std

        # self.poca_nn = poca_nn
        self.use_lstm = use_lstm

        self.hidden_layers = hidden_layers

        self.offsets = tf.stack(tf.meshgrid(
            tf.range(point_size, dtype=tf.int64),
            tf.range(point_size, dtype=tf.int64),
            tf.range(point_size, dtype=tf.int64),
        ), axis=-1)
        self.offsets -= tf.cast(tf.floor(point_size / 2), tf.int64)
        self.offsets = tf.reshape(self.offsets, (-1, 3))

        self.channel_indices = tf.range(channels, dtype=tf.int64)
        self.channel_indices = tf.repeat(self.channel_indices, point_size ** 3)[..., tf.newaxis]

        if projection_dim:
            self.projection = MLP(
                [projection_dim * k for k in hidden_layers],
                ['gelu', 'gelu'],
                name=f'{self.name}/projection'
            )

            self.lstm = Bidirectional(LSTM(projection_dim * 4, return_sequences=True))
            self.final_projection = Dense(projection_dim, name=f'out_projection')

        self.pointwise_conv = Dense(self.channels)
        self.poca_correction = Dense(3, activation="sigmoid")

    def call(self, inputs, *args, **kwargs):
        positions, x = inputs
        b = tf.shape(x)[0]
        s = tf.shape(x)[1]

        if self.projection_dim:
            x = self.projection(x)
            if self.use_lstm:
                x = self.lstm(x)
            x = self.final_projection(x)

        corrections = 0.1 * self.poca_correction(x)
        positions = positions + corrections

        positions = positions * self.resolution
        offsets = positions - tf.cast(tf.math.rint(positions), dtype=tf.float32)

        offsets = offsets[..., tf.newaxis, :]
        offsets = tf.repeat(offsets, repeats=self.point_size ** 3, axis=-2)
        offsets = offsets + tf.cast(self.offsets, dtype=tf.float32)
        offsets = offsets[..., tf.newaxis, :]
        offsets = tf.repeat(offsets, repeats=self.channels, axis=-2)

        positions = tf.cast(positions, tf.int64)[..., tf.newaxis, :]

        positions = tf.repeat(positions, repeats=self.point_size ** 3, axis=-2)
        positions = tf.clip_by_value(positions + self.offsets, 0, self.resolution - 1)
        positions = tf.repeat(positions, repeats=self.channels, axis=-2)

        # funny stuff to build a sparse tensor
        # index for batch dim and detections dim
        batch_indices = tf.range(b, dtype=tf.int64)
        batch_indices = tf.reshape(batch_indices, (-1, 1, 1, 1))
        batch_indices = tf.repeat(batch_indices, repeats=positions.shape[2], axis=2)
        batch_indices = tf.repeat(batch_indices, repeats=s, axis=1)

        indices = tf.concat([batch_indices, positions], axis=-1)

        channel_indices = tf.broadcast_to(self.channel_indices, (b, s, indices.shape[2], 1))
        indices = tf.concat([indices, channel_indices], axis=-1)

        indices = tf.reshape(indices, (-1, indices.shape[-1]))

        # applying gaussian
        gaussian = tf.exp(-tf.square(tf.norm(offsets, axis=-1) / self.std))
        x = tf.einsum("bdy,bdxy->bdxy", x, gaussian)
        x = tf.reshape(x, (-1, self.channels * self.point_size ** 3))

        features = tf.reshape(x, (-1,))

        x = tf.scatter_nd(indices, features,
                          shape=(b, self.resolution, self.resolution, self.resolution, self.channels))

        counts = tf.scatter_nd(indices, tf.ones_like(features),
                               shape=(b, self.resolution, self.resolution, self.resolution, self.channels))

        return self.pointwise_conv(tf.concat([x, counts[..., 0:1]], axis=-1))


if __name__ == "__main__":
    from layers import ScatterAndAvg3D

    a = GaussianScatterAndAvg3D(64, 8, projection_dim=8, point_size=5)
    b = ScatterAndAvg3D(64, 8, projection_dim=8, point_size=1)

    lst = [tf.random.normal((1, 10, 3)), tf.random.normal((1, 10, 12))]
    output = a(lst)
    output2 = b(lst)
