import tensorflow as tf

from keras.layers import *
from keras.models import *
from layers.convnext_block import ConvNeXtBlock

import numpy as np

_2d_base_params = {
    'resolution': 256,
    'scatter_filters': [[1, 9], [9, 9]],
    'mlp_units': [32 * 4, 32],
    'mlp_activations': ['gelu', 'linear'],
    'downward_blocks': [1, 1, 2, 3, 5],
    'downward_filters': [4, 32, 128, 512, 512],
    'upward_blocks': [5, 2, 1, 1],
    'upward_filters': [512, 256, 256, 256],
}


class MLP(Layer):
    def __init__(self, units, activations, dropout=0.0, **kwargs):
        super(MLP, self).__init__(**kwargs)

        assert len(units) == len(activations)

        self.units = units
        self.activations = activations

        self.layers = [
            Sequential(
                [
                    Dense(units=units[i], activation=activations[i], name=f'{self.name}-dense_{i}'),
                    Dropout(dropout, name=f'{self.name}-drop_{i}')
                ]
            )
            for i in range(len(units))
        ]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
        cfg = super(MLP, self).get_config()
        cfg.update({
            'units': self.units,
            'activations': self.activations
        })


class Agg2D(Model):
    def __init__(
            self,
            resolution=None,
            scatter_filters=None,
            mlp_units=None,
            mlp_activations=None,
            downward_blocks=None,
            downward_filters=None,
            upward_blocks=None,
            upward_filters=None,
            *args, **kwargs):
        super(Agg2D, self).__init__(*args, **kwargs)

        assert (len(scatter_filters) * 2 + sum([sum(x) for x in scatter_filters]) == mlp_units,
                f'{mlp_units} features cannot be scattered into the filters {scatter_filters}')

        self.resolution = resolution
        self.scatter_filters = scatter_filters

        self.mlp = MLP(mlp_units, mlp_activations)

        channel_indices = tf.expand_dims(tf.range(sum([len(x) for x in scatter_filters]), dtype=tf.int64), -1)
        self.channel_indices = tf.repeat(channel_indices, [c for lst in scatter_filters for c in lst], axis=-2)
        offsets = []
        for j in scatter_filters:
            for i in j:
                offsets.append(tf.stack(tf.meshgrid(
                    tf.range(int(np.sqrt(i)), dtype=tf.int64),
                    tf.range(int(np.sqrt(i)), dtype=tf.int64)
                ), axis=-1))
                offsets[-1] = tf.reshape(offsets[-1], (-1, 2))
                offsets[-1] -= int(np.floor(int(np.sqrt(i)) / 2))
        self.offsets = tf.concat(offsets, axis=0)

        self.downward_convs = []
        self.downsampling = []
        self.upward_convs = []
        self.upsampling = []
        self.upsampling_convs = []

        for stage in range(len(downward_blocks)):
            stack = []
            for c in range(downward_blocks[stage]):
                stack.append(ConvNeXtBlock(projection_dim=downward_filters[stage],
                                           dims=2,
                                           name=f'{self.name}-stage_{stage}-block_{c}'))
            self.downward_convs.append(Sequential(stack, name=f'{self.name}-stage_{stage}'))

        for stage in range(len(downward_blocks) - 1):
            self.downsampling.append(Sequential([
                LayerNormalization(epsilon=1e-6, name=f'{self.name}-stage_{stage}-downsampling-layer_norm'),
                Conv2D(filters=downward_filters[stage + 1],
                       kernel_size=2,
                       strides=2,
                       name=f'{self.name}-stage_{stage}-downsampling-conv2d',
                       padding='same')
            ], name=f'{self.name}-stage_{stage}-downsampling'))

            stack = []
            for c in range(upward_blocks[stage]):
                stack.append(ConvNeXtBlock(projection_dim=upward_filters[stage],
                                           dims=2,
                                           name=f'{self.name}-up_{stage}-block_{c}'))
            self.upward_convs.append(Sequential(stack, name=f'{self.name}-up_{stage}'))

            self.upsampling.append(Sequential([
                LayerNormalization(epsilon=1e-6, name=f'{self.name}-up_{stage}-upsampling-layer_norm'),
                UpSampling2D(size=(2, 2), name=f'{self.name}-up_{stage}-upsampling-upsampling'),
            ], name=f'{self.name}-up_{stage}-upsampling'))

            self.upsampling_convs.append(
                Conv2D(filters=upward_filters[stage],
                       kernel_size=1,
                       padding='same',
                       name=f'{self.name}-up_{stage}-upsampling-conv2d')
            )

    def call(self, inputs, training=None, mask=None):
        b = tf.shape(inputs)[0]
        n = inputs.shape[1]
        features = self.mlp(inputs)

        # funny stuff to build a sparse tensor
        # index for batch dim and detections dim
        batch_indices = tf.stack(tf.meshgrid(
            tf.range(b, dtype=tf.int64),
            tf.range(n, dtype=tf.int64),
            indexing='ij',
        ), axis=-1,)

        # index for height and width
        # the position is encoded as part of the feature vector
        # `scatter_filters` is a nested list. each inner list corresponds to one position.
        # each element in the inner list corresponds to the feature map size for one channel.
        # positions = tf.reshape(
        #     features[..., :len(self.scatter_filters) * 2],
        #     (-1, inputs.shape[1], len(self.scatter_filters), 2)
        # )

        positions = tf.reshape(
            tf.gather(inputs, [1, 2, 7, 8], axis=-1) / 2 + 1,
            (-1, inputs.shape[1], len(self.scatter_filters), 2)
        )

        positions = tf.repeat(
            positions,
            repeats=[len(x) for x in self.scatter_filters],
            axis=-2
        ) * self.resolution  # for each sublist

        positions = tf.cast(positions, tf.int64)
        positions = tf.repeat(
            positions,
            repeats=[c for lst in self.scatter_filters for c in lst], axis=-2
        )  # for each element

        # each point in the feature map is offset from the center by some value
        positions = tf.clip_by_value(
            positions + self.offsets,
            0 + int(np.sqrt(max([max(lst) for lst in self.scatter_filters])) / 2),
            self.resolution - int(np.sqrt(max([max(lst) for lst in self.scatter_filters])) / 2)
        )

        batch_indices = batch_indices[:, :, tf.newaxis, :]
        batch_indices = tf.repeat(batch_indices, repeats=(positions.shape[2]), axis=2)

        indices = tf.concat([batch_indices, positions], axis=-1)

        # indices for channel dim
        channel_indices = tf.broadcast_to(self.channel_indices,
                                          (tf.shape(indices)[0], indices.shape[1], indices.shape[2], 1))
        indices = tf.concat([indices, channel_indices], axis=-1)

        features = tf.reshape(features, (-1,))

        indices = tf.reshape(indices, (-1, tf.shape(indices)[-1]))

        x = tf.sparse.SparseTensor(
            indices=indices,
            values=features,
            dense_shape=[b, n, self.resolution, self.resolution, sum([len(lst) for lst in self.scatter_filters])]
        )

        x = tf.sparse.reduce_sum(x, axis=1) / tf.cast(n, tf.float32)
        x.set_shape([inputs.shape[0], self.resolution, self.resolution,
                     sum([len(lst) for lst in self.scatter_filters])])

        skip_outputs = []
        for i, block in enumerate(self.downward_convs):
            x = block(x)
            if i != len(self.downward_convs) - 1:
                skip_outputs.append(x)
                x = self.downsampling[i](x)

        for i, block in enumerate(self.upward_convs):
            x = self.upsampling[i](x)
            x = tf.concat([x, skip_outputs.pop()], axis=-1)
            x = self.upsampling_convs[i](x)
            x = block(x)

        return x


if __name__ == "__main__":
    test = Agg2D(**_2d_base_params)
    print(test(tf.ones((1, 8096, 4))).shape)
