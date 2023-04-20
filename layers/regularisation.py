import tensorflow as tf
keras = tf.keras

from keras import backend as K
from keras.layers import *


class StochasticDepth(Layer):
    """
    Implements the Stochastic Depth layer. It randomly drops residual branches
    in residual architectures. It is used as a drop-in replacement for addition
    operation. Note that this layer DOES NOT drop a residual block across
    individual samples but across the entire batch.
    Reference:
        - [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
        - Docstring taken from [stochastic_depth.py](https://tinyurl.com/mr3y2af6)
    Args:
        rate: float, the probability of the residual branch being dropped.
    Usage:
    `StochasticDepth` can be used in a residual network as follows:
    ```python
    # (...)
    input = tf.ones((1, 3, 3, 1), dtype=tf.float32)
    residual = tf.keras.layers.Conv2D(1, 1)(input)
    output = keras_cv.layers.StochasticDepth()([input, residual])
    # (...)
    ```
    At train time, StochasticDepth returns:
    $$
    x[0] + b_l * x[1],
    $$
    where $b_l$ is a random Bernoulli variable with probability
    $P(b_l = 1) = rate$. At test time, StochasticDepth rescales the activations
    of the residual branch based on the drop rate ($rate$):
    $$
    x[0] + (1 - rate) * x[1]
    $$
    """

    def __init__(self, rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.survival_probability = 1.0 - self.rate

    def call(self, x, training=None, **kwargs):
        if len(x) != 2:
            raise ValueError(
                f"""Input must be a list of length 2. """
                f"""Got input with length={len(x)}."""
            )

        shortcut, residual = x

        b_l = K.random_bernoulli([], p=self.survival_probability)

        if training:
            return shortcut + b_l * residual
        else:
            return shortcut + self.survival_probability * residual

    def get_config(self):
        config = {"rate": self.rate}
        base_config = super().get_config()
        return base_config.update(config)


class DropBlock1D(Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock1D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.supports_masking = True
        self.seq_len = self.ones = self.zeros = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            self.seq_len = input_shape[-1]
        else:
            self.seq_len = input_shape[1]
        self.ones = K.ones(self.seq_len, name='ones')
        self.zeros = K.zeros(self.seq_len, name='zeros')
        super().build(input_shape)

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock1D, self).get_config()
        return base_config.update(config)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self):
        """Get the number of activation units to drop"""
        feature_dim = K.cast(self.seq_len, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / block_size) * (feature_dim / (feature_dim - block_size + 1.0))

    def _compute_valid_seed_region(self):
        positions = K.arange(self.seq_len)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions >= half_block_size,
                        positions < self.seq_len - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            self.ones,
            self.zeros,
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        mask = K.random_binomial(shape, p=self._get_gamma())
        mask *= self._compute_valid_seed_region()
        mask = MaxPool1D(
            pool_size=self.block_size,
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None, **kwargs):

        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = outputs * mask * \
                      (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 1])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


class DropBlock2D(Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.supports_masking = True
        self.height = self.width = self.ones = self.zeros = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            self.height, self.width = input_shape[2], input_shape[3]
        else:
            self.height, self.width = input_shape[1], input_shape[2]
        self.ones = K.ones((self.height, self.width), name='ones')
        self.zeros = K.zeros((self.height, self.width), name='zeros')
        super().build(input_shape)

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock2D, self).get_config()
        return base_config.update(config)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self):
        """Get the number of activation units to drop"""
        height, width = K.cast(self.height, K.floatx()), K.cast(self.width, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / (block_size ** 2)) * \
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

    def _compute_valid_seed_region(self):
        positions = K.concatenate([
            K.expand_dims(K.tile(K.expand_dims(K.arange(self.height), axis=1), [1, self.width]), axis=-1),
            K.expand_dims(K.tile(K.expand_dims(K.arange(self.width), axis=0), [self.height, 1]), axis=-1),
        ], axis=-1)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions[:, :, 0] >= half_block_size,
                        positions[:, :, 1] >= half_block_size,
                        positions[:, :, 0] < self.height - half_block_size,
                        positions[:, :, 1] < self.width - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            self.ones,
            self.zeros,
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        mask = K.random_binomial(shape, p=self._get_gamma())
        mask *= self._compute_valid_seed_region()
        mask = MaxPool2D(
            pool_size=(self.block_size, self.block_size),
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None, **kwargs):
        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = outputs * mask * \
                      (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


class DropBlock3D(Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock3D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.supports_masking = True
        self.height = self.width = self.depth = self.ones = self.zeros = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            self.height, self.width, self.depth = input_shape[2], input_shape[3], input_shape[4]
        else:
            self.height, self.width, self.depth = input_shape[1], input_shape[2], input_shape[3]

        self.ones = K.ones((self.height, self.width, self.depth), name='ones')
        self.zeros = K.zeros((self.height, self.width, self.depth), name='zeros')
        super().build(input_shape)

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock3D, self).get_config()
        return base_config.update(config)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self):
        """Get the number of activation units to drop"""
        height, width, depth = K.cast(self.height, K.floatx()), \
                               K.cast(self.width, K.floatx()), K.cast(self.depth, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / (block_size ** 2)) * \
               (height * width * depth / ((height - block_size + 1.0) * (width - block_size + 1.0) *
                                          (depth - block_size + 1.0)))

    def _compute_valid_seed_region(self):
        positions = []
        for i in range(self.height):
            positions.append([])
            for j in range(self.width):
                positions[-1].append([])
                for k in range(self.depth):
                    positions[-1][-1].append([i, j, k])

        positions = tf.constant(positions)

        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions[:, :, :, 0] >= half_block_size,
                        positions[:, :, :, 1] >= half_block_size,
                        positions[:, :, :, 2] >= half_block_size,
                        positions[:, :, :, 0] < self.height - half_block_size,
                        positions[:, :, :, 1] < self.width - half_block_size,
                        positions[:, :, :, 2] < self.depth - half_block_size
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            self.ones,
            self.zeros,
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        mask = K.random_binomial(shape, p=self._get_gamma())
        mask *= self._compute_valid_seed_region()
        mask = MaxPool3D(
            pool_size=(self.block_size, self.block_size, self.block_size),
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)

        return 1.0 - mask

    def call(self, inputs, training=None, **kwargs):
        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 3, 4, 1])

            shape = K.shape(outputs)

            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], shape[2], shape[3], 1])
            else:
                mask = self._compute_drop_mask(shape)

            outputs = outputs * mask * \
                      (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))

            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 4, 1, 2, 3])

            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


def test():
    height = 200
    width = 100

    positions = K.concatenate([
        K.expand_dims(K.tile(K.expand_dims(K.arange(height), axis=1), [1, width]), axis=-1),
        K.expand_dims(K.tile(K.expand_dims(K.arange(width), axis=0), [height, 1]), axis=-1),
    ], axis=-1)
    print(positions)

    positions = []
    for i in range(height):
        positions.append([])
        for j in range(width):
            positions[-1].append([])
            for k in range(200):
                positions[-1][-1].append([i, j, k])

    positions = tf.constant(positions)
    print(positions)


if __name__ == "__main__":
    test()
