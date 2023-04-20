import tensorflow as tf
keras = tf.keras

from keras.layers import *
from keras.models import Sequential


class SqueezeAndExcite2D(Layer):
    """
    Implements Squeeze and Excite block as in
    [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf).
    This layer tries to use a content aware mechanism to assign channel-wise
    weights adaptively. It first squeezes the feature maps into a single value
    using global average pooling, which are then fed into two Conv1D layers,
    which act like fully-connected layers. The first layer reduces the
    dimensionality of the feature maps by a factor of `ratio`, whereas the second
    layer restores it to its original value.
    The resultant values are the adaptive weights for each channel. These
    weights are then multiplied with the original inputs to scale the outputs
    based on their individual weightages.
    Args:
        filters: Number of input and output filters. The number of input and
            output filters is same.
        ratio: Ratio for bottleneck filters. Number of bottleneck filters =
            filters * ratio. Defaults to 0.25.
        squeeze_activation: (Optional) String, callable (or tf.keras.layers.Layer) or
            tf.keras.activations.Activation instance denoting activation to
            be applied after squeeze convolution. Defaults to `relu`.
        excite_activation: (Optional) String, callable (or tf.keras.layers.Layer) or
            tf.keras.activations.Activation instance denoting activation to
            be applied after excite convolution. Defaults to `sigmoid`.
    Usage:
    ```python
    # (...)
    input = tf.ones((1, 5, 5, 16), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(16, (3, 3))(input)
    output = keras_cv.layers.SqueezeAndExciteBlock(16)(x)
    # (...)
    ```
    """

    def __init__(
        self,
        filters,
        ratio=0.25,
        squeeze_activation="relu",
        excite_activation="sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters

        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(f"`ratio` should be a float between 0 and 1. Got {ratio}")

        if filters <= 0 or not isinstance(filters, int):
            raise ValueError(f"`filters` should be a positive integer. Got {filters}")

        self.ratio = ratio
        self.bottleneck_filters = int(self.filters * self.ratio)

        self.squeeze_activation = squeeze_activation
        self.excite_activation = excite_activation

        self.global_average_pool = GlobalAveragePooling2D()
        self.reshape = Reshape((1, 1, filters))
        self.squeeze_conv = Conv2D(
            self.bottleneck_filters, (1, 1),
            activation=self.squeeze_activation,
        )
        self.excite_conv = Conv2D(
            self.filters, (1, 1), activation=self.excite_activation
        )

    def call(self, inputs, training=True):
        x = self.global_average_pool(inputs)
        x = self.reshape(x)  # x: (batch_size, 1, 1, filters)
        x = self.squeeze_conv(x)  # x: (batch_size, 1, 1, bottleneck_filters)
        x = self.excite_conv(x)  # x: (batch_size, 1, 1, filters)
        x = tf.math.multiply(x, inputs)  # x: (batch_size, h, w, filters)
        return x

    def get_config(self):
        config = {
            "filters": self.filters,
            "ratio": self.ratio,
            "squeeze_activation": self.squeeze_activation,
            "excite_activation": self.excite_activation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SqueezeAndExcite3D(Layer):
    """
    Implements Squeeze and Excite block as in
    [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf).
    This layer tries to use a content aware mechanism to assign channel-wise
    weights adaptively. It first squeezes the feature maps into a single value
    using global average pooling, which are then fed into two Conv1D layers,
    which act like fully-connected layers. The first layer reduces the
    dimensionality of the feature maps by a factor of `ratio`, whereas the second
    layer restores it to its original value.
    The resultant values are the adaptive weights for each channel. These
    weights are then multiplied with the original inputs to scale the outputs
    based on their individual weightages.
    Args:
        filters: Number of input and output filters. The number of input and
            output filters is same.
        ratio: Ratio for bottleneck filters. Number of bottleneck filters =
            filters * ratio. Defaults to 0.25.
        squeeze_activation: (Optional) String, callable (or tf.keras.layers.Layer) or
            tf.keras.activations.Activation instance denoting activation to
            be applied after squeeze convolution. Defaults to `relu`.
        excite_activation: (Optional) String, callable (or tf.keras.layers.Layer) or
            tf.keras.activations.Activation instance denoting activation to
            be applied after excite convolution. Defaults to `sigmoid`.
    Usage:
    ```python
    # (...)
    input = tf.ones((1, 5, 5, 16), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(16, (3, 3))(input)
    output = keras_cv.layers.SqueezeAndExciteBlock(16)(x)
    # (...)
    ```
    """

    def __init__(
        self,
        filters,
        ratio=0.25,
        squeeze_activation="relu",
        excite_activation="sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters

        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(f"`ratio` should be a float between 0 and 1. Got {ratio}")

        if filters <= 0 or not isinstance(filters, int):
            raise ValueError(f"`filters` should be a positive integer. Got {filters}")

        self.ratio = ratio
        self.bottleneck_filters = int(self.filters * self.ratio)

        self.squeeze_activation = squeeze_activation
        self.excite_activation = excite_activation

        self.global_average_pool = GlobalAveragePooling3D()
        self.reshape = Reshape((1, 1, 1, filters))
        self.squeeze_conv = Conv3D(
            self.bottleneck_filters, (1, 1, 1),
            activation=self.squeeze_activation,
        )
        self.excite_conv = Conv3D(
            self.filters, (1, 1, 1), activation=self.excite_activation
        )

    def call(self, inputs, training=True):
        x = self.global_average_pool(inputs)
        x = self.reshape(x)  # x: (batch_size, 1, 1, filters)
        x = self.squeeze_conv(x)  # x: (batch_size, 1, 1, bottleneck_filters)
        x = self.excite_conv(x)  # x: (batch_size, 1, 1, filters)
        x = tf.math.multiply(x, inputs)  # x: (batch_size, h, w, filters)
        return x

    def get_config(self):
        config = {
            "filters": self.filters,
            "ratio": self.ratio,
            "squeeze_activation": self.squeeze_activation,
            "excite_activation": self.excite_activation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ChannelwiseMaxPooling2D(Layer):
    def __init__(self, **kwargs):
        super(ChannelwiseMaxPooling2D, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.expand_dims(tf.math.reduce_max(inputs, axis=-1), axis=-1)


class ChannelwiseAveragePooling2D(Layer):
    def __init__(self, **kwargs):
        super(ChannelwiseAveragePooling2D, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.expand_dims(tf.math.reduce_mean(inputs, axis=-1), axis=-1)


class SpatialAttentionModule(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionModule, self).__init__(**kwargs)

        self.global_max = GlobalMaxPooling2D()
        self.global_average = GlobalAveragePooling2D()

        self.sigmoid = Activation("sigmoid")

        self.channel_max = ChannelwiseMaxPooling2D()
        self.channel_average = ChannelwiseAveragePooling2D()

    def build(self, input_shape):
        self.mlp = Sequential([
            Dense(256, activation="swish"),
            Dense(input_shape[-1])
        ])

        self.conv2d = Conv2D(input_shape[-1], 9, activation="sigmoid", padding="same")

    def call(self, input_tensor):
        # Channel Attention
        x1 = self.global_max(input_tensor)
        x2 = self.global_average(input_tensor)

        x = add([self.mlp(x1), self.mlp(x2)])
        x = self.sigmoid(x)
        x = multiply([x, input_tensor])

        # Spatial Attention
        x1 = self.channel_max(x)
        x2 = self.channel_average(x)

        x = concatenate([x1, x2])
        x = self.conv2d(x)
        return multiply([x, input_tensor])


def global_context_block(reduction_ratio=16, name=None):  # stopped being lazy and implemented it myself
    def apply(x):
        # Context Modelling
        context = Dense(1, activation="softmax", name=f"{name}_context_conv1x1")(x)
        context = Reshape((1, 1, x.shape[1] * x.shape[2]), name=f"{name}_reshape_1")(context)
        context = dot([context, Reshape((x.shape[1] * x.shape[2], x.shape[-1]), name=f"{name}_reshape_2")(x)], axes=(3, 1))

        # Transform
        transform = Dense(x.shape[-1] // reduction_ratio, name=f"{name}_conv1x1_transform_1")(context)
        transform = LayerNormalization(epsilon=1e-6, name=f"{name}_layernorm")(transform)
        transform = Activation("swish", name=f"{name}_swish")(transform)
        transform = Dense(x.shape[-1], name=f"{name}_conv1x1_transform_2")(transform)

        return x + transform

    return apply


"""
# Stolen from https://github.com/titu1994/keras-global-context-networks/blob/master/gc.py
def global_context_block(ip, reduction_ratio=16, transform_activation='linear'):
    Adds a Global Context attention block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).
    # Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        reduction_ratio: Reduces the input filters by this factor for the
            bottleneck block of the transform submodule. Node: the reduction
            ratio must be set such that it divides the input number of channels,
        transform_activation: activation function to apply to the output
            of the transform block. Can be any string activation function availahle
            to Keras.
    # Returns:
        a tensor of same shape as input
    channel_dim = -1
    ip_shape = ip.shape

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    if rank > 3:
        flat_spatial_dim = -1
    else:
        flat_spatial_dim = 1

    Context Modelling Block
    # [B, ***, C] or [B, C, ***]
    input_flat = _spatial_flattenND(ip, rank)
    # [B, ..., C] or [B, C, ...]
    context = _convND(ip, rank, channels=1, kernel=1)
    # [B, ..., 1] or [B, 1, ...]
    context = _spatial_flattenND(context, rank)
    # [B, ***, 1] or [B, 1, ***]
    context = Softmax(axis=flat_spatial_dim)(context)

    # Compute context block outputs
    context = dot([input_flat, context], axes=1)
    # [B, C, 1]
    context = _spatial_expandND(context, rank)
    # [B, C, 1...] or [B, 1..., C]

    Transform block
    # Transform bottleneck
    # [B, C // R, 1...] or [B, 1..., C // R]
    transform = _convND(context, rank, channels // reduction_ratio, kernel=1)
    # Group normalization acts as Layer Normalization when groups = 1
    transform = GroupNormalization(groups=1, axis=channel_dim)(transform)
    transform = Activation("swish")(transform)

    # Transform output block
    # [B, C, 1...] or [B, 1..., C]
    transform = _convND(transform, rank, channels, kernel=1)
    transform = Activation(transform_activation)(transform)

    # apply context transform
    out = add([ip, transform])

    return out


def _convND(ip, rank, channels, kernel=1):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, kernel, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (kernel, kernel), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (kernel, kernel, kernel), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)

    return x


def _spatial_flattenND(ip, rank):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    ip_shape = ip.shape
    channel_dim = -1

    if rank == 3:
        x = ip  # identity op for rank 3

    elif rank == 4:
        if channel_dim == 1:
            # [C, D1, D2] -> [C, D1 * D2]
            shape = [ip_shape[1], ip_shape[2] * ip_shape[3]]
        else:
            # [D1, D2, C] -> [D1 * D2, C]
            shape = [ip_shape[1] * ip_shape[2], ip_shape[3]]

        x = Reshape(shape)(ip)

    else:
        if channel_dim == 1:
            # [C, D1, D2, D3] -> [C, D1 * D2 * D3]
            shape = [ip_shape[1], ip_shape[2] * ip_shape[3] * ip_shape[4]]
        else:
            # [D1, D2, D3, C] -> [D1 * D2 * D3, C]
            shape = [ip_shape[1] * ip_shape[2] * ip_shape[3], ip_shape[4]]

        x = Reshape(shape)(ip)

    return x


def _spatial_expandND(ip, rank):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    channel_dim = -1

    if rank == 3:
        x = Permute((2, 1))(ip)  # identity op for rank 3

    elif rank == 4:
        if channel_dim == 1:
            # [C, D1, D2] -> [C, D1 * D2]
            shape = [-1, 1, 1]
        else:
            # [D1, D2, C] -> [D1 * D2, C]
            shape = [1, 1, -1]

        x = Reshape(shape)(ip)

    else:
        if channel_dim == 1:
            # [C, D1, D2, D3] -> [C, D1 * D2 * D3]
            shape = [-1, 1, 1, 1]
        else:
            # [D1, D2, D3, C] -> [D1 * D2 * D3, C]
            shape = [1, 1, 1, -1]

        x = Reshape(shape)(ip)

    return x
"""
