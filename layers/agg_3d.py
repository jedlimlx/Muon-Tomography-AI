import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers.poca import poca
from layers.agg_2d import MLP
from layers.convnext_block import ConvNeXtBlock
from layers.residual_block import ResidualBlock


class ScatterAndAvg3D(Layer):
    def __init__(self, resolution, channels, hidden_layers=(8, 4),
                 point_size=3, projection_dim=None, poca_nn=False, use_lstm=False, **kwargs):
        super().__init__(**kwargs)
        self.resolution = resolution
        self.point_size = point_size
        self.projection_dim = projection_dim
        self.channels = channels

        self.poca_nn = poca_nn
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

        self.pointwise_conv = Dense(projection_dim)

    def call(self, inputs, *args, **kwargs):
        positions, x = inputs
        b = tf.shape(x)[0]
        s = tf.shape(x)[1] * (1 if not self.poca_nn else 5)

        if self.projection_dim:
            x = self.projection(x)
            if self.use_lstm:
                x = self.lstm(x)
            x = self.final_projection(x)

        positions = positions * self.resolution
        positions = tf.cast(positions, tf.int64)[..., tf.newaxis, :]
        positions = tf.repeat(positions, repeats=self.point_size ** 3, axis=-2)
        positions = tf.clip_by_value(positions + self.offsets, 0, self.resolution - 1)
        positions = tf.repeat(positions, repeats=self.channels, axis=-2)

        # funny stuff to build a sparse tensor
        # index for batch dim and detections dim
        batch_indices = tf.range(b, dtype=tf.int64)
        batch_indices = tf.reshape(batch_indices, (-1, 1, 1, 1))
        batch_indices = tf.repeat(batch_indices, repeats=(positions.shape[2]), axis=2)
        batch_indices = tf.repeat(batch_indices, repeats=s, axis=1)

        indices = tf.concat([batch_indices, positions], axis=-1)

        channel_indices = tf.broadcast_to(self.channel_indices, (b, s, indices.shape[2], 1))
        indices = tf.concat([indices, channel_indices], axis=-1)

        indices = tf.reshape(indices, (-1, indices.shape[-1]))

        features = tf.reshape(x, (-1,))

        x = tf.scatter_nd(indices, features,
                          shape=(b, self.resolution, self.resolution, self.resolution, self.channels))

        counts = tf.scatter_nd(indices, tf.ones_like(features),
                               shape=(b, self.resolution, self.resolution, self.resolution, self.channels))

        return self.pointwise_conv(tf.concat([x, counts[..., 0:1]], axis=-1))  # tf.math.divide_no_nan(x, counts)
