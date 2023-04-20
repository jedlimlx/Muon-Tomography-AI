import tensorflow as tf
keras = tf.keras

from keras.layers import *

from layers.regularisation import StochasticDepth
from layers.attention import SqueezeAndExcite2D, SqueezeAndExcite3D, SpatialAttentionModule, global_context_block

from keras_cv_attention_models.attention_layers import mhsa_with_multi_head_relative_position_embedding


class LayerScale(Layer):
    """Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.
    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(self.init_values * tf.ones((self.projection_dim,)))

    def call(self, x, **kwargs):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


def ConvNeXtBlock(
        projection_dim,
        drop_connect_rate=0.0,
        layer_scale_init_value=1e-6,
        activation="gelu",
        kernel_size=7,
        attention="se",
        dims=2,
        name=None
):
    """ConvNeXt block.
    References:
      - https://arxiv.org/abs/2201.03545
      - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    Notes:
      In the original ConvNeXt implementation (linked above), the authors use
      `Dense` layers for pointwise convolutions for increased efficiency.
      Following that, this implementation also uses the same.
    Args:
      projection_dim (int): Number of filters for convolution layers. In the
        ConvNeXt paper, this is referred to as projection dimension.
      drop_connect_rate (float): Probability of dropping paths. Should be within
        [0, 1].
      layer_scale_init_value (float): Layer scale value. Should be a small float
        number.
      activation (string): activation function for the ConvNeXt block
      kernel_size (int): kernel size of the convolution
      dims (int): Number of dimensions for the block. Either 2 or 3
      attention (string): attention module to use
      name: name to path to the keras layer.
    Returns:
      A function representing a ConvNeXtBlock block.
    """

    se_block = SqueezeAndExcite2D if dims == 2 else SqueezeAndExcite3D
    cbam_block = SpatialAttentionModule

    def apply(inputs):
        x = inputs

        if dims == 1:
            x = Conv1D(
                filters=projection_dim,
                kernel_size=kernel_size,
                padding="same",
                groups=projection_dim,
                name=name + "_depthwise_conv",
            )(x)
        elif dims == 2:
            x = Conv2D(
                filters=projection_dim,
                kernel_size=kernel_size,
                padding="same",
                groups=projection_dim,
                name=name + "_depthwise_conv",
            )(x)
        elif dims == 3:
            x = Conv3D(
                filters=projection_dim,
                kernel_size=kernel_size,
                padding="same",
                groups=projection_dim,
                name=name + "_depthwise_conv",
            )(x)

        x = LayerNormalization(epsilon=1e-6, name=name + "_layernorm")(x)
        x = Dense(4 * projection_dim, name=name + "_pointwise_conv_1")(x)
        x = Activation(activation, name=name + "_" + activation)(x)
        x = Dense(projection_dim, name=name + "_pointwise_conv_2")(x)

        if layer_scale_init_value is not None:
            x = LayerScale(
                layer_scale_init_value,
                projection_dim,
                name=name + "_layer_scale",
            )(x)

        if attention is not None:
            if attention == "se":
                x = se_block(projection_dim, name=name + "_squeeze_exite")(x)
            elif attention == "cbam":
                x = cbam_block()(x)
            elif attention == "gc" and dims == 2:
                x = global_context_block(name=f"{name}_gc")(x)
            elif attention == "coatnet" and dims == 2:
                x = mhsa_with_multi_head_relative_position_embedding(x)

        if drop_connect_rate:
            layer = StochasticDepth(drop_connect_rate, name=name + "_stochastic_depth")
            return layer([inputs, x])
        else:
            layer = Activation("linear", name=name + "_identity")
            return inputs + layer(x)

    return apply
