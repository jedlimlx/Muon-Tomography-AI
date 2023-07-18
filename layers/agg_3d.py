import tensorflow as tf
from keras.layers import *
from keras.models import *
from layers.convnext_block import ConvNeXtBlock
from layers.poca import poca
from layers.agg_2d import MLP


_3d_base_params = {
    'point_size': 3,
    'downward_convs': [1, 1, 2, 3, 5],
    'downward_filters': [4, 16, 64, 128, 256],
    'upward_convs': [4, 3, 2, 1],
    'upward_filters': [128, 64, 16, 4],
    'resolution': 64,
}


class Agg3D(Model):
    def __init__(self,
                 point_size=3,
                 downward_convs=None,
                 downward_filters=None,
                 upward_convs=None,
                 upward_filters=None,
                 resolution=None,
                 *args, **kwargs):
        super(Agg3D, self).__init__(*args, **kwargs)
        self.resolution = resolution
        self.point_size = point_size
        self.downward_filters = downward_filters

        self.mlp = MLP(
            [point_size ** 3 * downward_filters[0] * 4, point_size ** 3 * downward_filters[0]],
            ['gelu', 'linear'],
        )

        # stuff for building the sparse tensor
        self.offsets = tf.stack(tf.meshgrid(
            tf.range(point_size, dtype=tf.int64),
            tf.range(point_size, dtype=tf.int64),
            tf.range(point_size, dtype=tf.int64),
        ), axis=-1)
        self.offsets -= tf.cast(tf.floor(point_size / 2), tf.int64)
        self.offsets = tf.reshape(self.offsets, (-1, 3))

        self.channel_indices = tf.range(downward_filters[0], dtype=tf.int64)
        self.channel_indices = tf.repeat(self.channel_indices, point_size ** 3)[..., tf.newaxis]

        # downward ConvNeXt blocks
        self.downward_convs = []
        for stage in range(len(downward_convs)):
            stack = []
            for c in range(downward_convs[stage]):
                stack.append(ConvNeXtBlock(
                    downward_filters[stage],
                    kernel_size=3,
                    dims=3,
                    name=f'{self.name}/downward_stage_{stage}/block_{c}'
                ))
            self.downward_convs.append(Sequential(stack, name=f'{self.name}/downward_stage_{stage}'))

        self.downsampling = []
        self.upsampling = []
        self.upward_convs = []
        for stage in range(len(downward_convs) - 1):
            # downsampling convolutions
            self.downsampling.append(Sequential([
                LayerNormalization(epsilon=1e-6, name=f'{self.name}/downward_stage_{stage}/downsampling/layer_norm'),
                Conv3D(filters=downward_filters[stage + 1],
                       kernel_size=2,
                       strides=2,
                       name=f'{self.name}/downward_stage_{stage}/downsampling/conv2d',
                       padding='same')
            ], name=f'{self.name}/downward_stage_{stage}/downsampling'))

            # upsampling + convolutions
            self.upsampling.append(Sequential([
                Dense(units=upward_filters[stage], name=f'{self.name}/upward_stage_{stage}/upsampling/pointwise_conv'),
                LayerNormalization(epsilon=1e-6, name=f'{self.name}/upward_stage_{stage}/upsampling/layer_norm'),
                UpSampling3D(name=f'{self.name}/upward_stage_{stage}/upsampling/upsampling'),
            ]))

            # upward ConvNeXt blocks
            stack = []
            for c in range(upward_convs[stage]):
                stack.append(ConvNeXtBlock(
                    upward_filters[stage],
                    kernel_size=3,
                    dims=3,
                    name=f'{self.name}/upward_stage_{stage}/block_{c}'
                ))
            self.upward_convs.append(Sequential(stack, name=f'{self.name}/upward_stage_{stage}'))

        self.final_conv = Conv3D(1, 1, name=f'{self.name}/final_conv')

    def call(self, inputs, training=None, mask=None):
        b = tf.shape(inputs)[0]
        n = inputs.shape[1]

        # data format of inputs is x, y, z, px, py, pz, ver_x, ver_y, ver_z, ver_px, ver_py, ver_pz
        positions = poca(*tf.split(inputs, 4, axis=-1)) * self.resolution
        positions = tf.cast(positions, tf.int64)[..., tf.newaxis, :]
        positions = tf.repeat(positions, repeats=self.point_size ** 3, axis=-2)
        positions = tf.clip_by_value(positions + self.offsets, 0, self.resolution - 1)
        positions = tf.repeat(positions, repeats=self.downward_filters[0], axis=-2)

        # funny stuff to build a sparse tensor
        # index for batch dim and detections dim
        batch_indices = tf.stack(tf.meshgrid(
            tf.range(b, dtype=tf.int64),
            tf.range(n, dtype=tf.int64),
            indexing='ij',
        ), axis=-1)
        batch_indices = batch_indices[..., tf.newaxis, :]
        batch_indices = tf.repeat(batch_indices, repeats=(positions.shape[2]), axis=2)

        indices = tf.concat([batch_indices, positions], axis=-1)

        channel_indices = tf.broadcast_to(self.channel_indices, (b, n, indices.shape[2], 1))
        indices = tf.concat([indices, channel_indices], axis=-1)
        indices = tf.reshape(indices, (-1, indices.shape[-1]))

        features = self.mlp(inputs)
        features = tf.reshape(features, (-1,))

        x = tf.sparse.SparseTensor(
            indices=indices,
            values=features,
            dense_shape=(b, n, self.resolution, self.resolution, self.resolution, self.downward_filters[0])
        )

        x = tf.sparse.reduce_sum(x, axis=1) / tf.cast(n, tf.float32)
        x.set_shape((inputs.shape[0], self.resolution, self.resolution, self.resolution, self.downward_filters[0]))

        skip_outputs = []
        for i, block in enumerate(self.downward_convs):
            x = block(x)
            if i != len(self.downsampling):
                skip_outputs.append(x)
                x = self.downsampling[i](x)

        for i, block in enumerate(self.upward_convs):
            x = self.upsampling[i](x)
            x += skip_outputs.pop()
            x = block(x)

        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    test = Agg3D(**_3d_base_params)
    print(test(tf.random.normal((1, 16384, 12))).shape)
    test.summary()
