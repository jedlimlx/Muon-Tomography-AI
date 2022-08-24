from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers.regularisation import StochasticDepth


def ResidualBlock(filters, kernel_size=3, stride=1, conv_shortcut=True, name=None,
                  activation="relu", drop_connect_rate=0.2):
    """A residual block.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
      activation: string, the activation function to use
      drop_connect_rate: string, the probability that the block will be skipped
    Returns:

      Output tensor for the residual block.
    """

    def apply(x):
        if conv_shortcut:
            shortcut = Conv2D(4 * filters, 1, strides=stride, use_bias=False, name=name + "_0_conv")(x)
            shortcut = BatchNormalization(name=name + "_0_bn")(shortcut)
        else:
            shortcut = x

        x = Conv2D(filters, 1, strides=stride, use_bias=False, name=name + "_1_conv")(x)
        x = BatchNormalization(name=name + "_1_bn")(x)
        x = Activation(activation, name=name + f"_1_{activation}")(x)

        x = Conv2D(filters, kernel_size, padding="SAME", use_bias=False, name=name + "_2_conv")(x)
        x = BatchNormalization(name=name + "_2_bn")(x)
        x = Activation(activation, name=name + f"_2_{activation}")(x)

        x = Conv2D(4 * filters, 1, use_bias=False, name=name + "_3_conv")(x)
        x = BatchNormalization(name=name + "_3_bn")(x)

        x = StochasticDepth(rate=drop_connect_rate, name=name + "_add")([shortcut, x])
        x = Activation(activation, name=name + "_out")(x)
        return x

    return apply


def ResidualStack(filters, blocks, stride=2, name=None, activation="relu", drop_connect_rate=0.2):
    """A set of stacked residual blocks.
    Args:
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride: default 2, stride of the first layer in the first block.
      name: string, stack label.
      activation: string, the activation function to use
      drop_connect_rate: string, the probability that the block will be skipped
    Returns:
      Output tensor for the stacked blocks.
    """

    def apply(x):
        x = ResidualBlock(filters, stride=stride, name=name + "_block1", activation=activation,
                          drop_connect_rate=drop_connect_rate)(x)
        for i in range(2, blocks + 1):
            x = ResidualBlock(filters, conv_shortcut=False, name=name + "_block" + str(i), activation=activation,
                              drop_connect_rate=drop_connect_rate)(x)

        return x

    return apply
