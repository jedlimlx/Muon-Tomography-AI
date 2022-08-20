from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def ResidualBlock(filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
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
        x = Activation("relu", name=name + "_1_relu")(x)

        x = Conv2D(filters, kernel_size, padding="SAME", use_bias=False, name=name + "_2_conv")(x)
        x = BatchNormalization(name=name + "_2_bn")(x)
        x = Activation("relu", name=name + "_2_relu")(x)

        x = Conv2D(4 * filters, 1, use_bias=False, name=name + "_3_conv")(x)
        x = BatchNormalization(name=name + "_3_bn")(x)

        x = Add(name=name + "_add")([shortcut, x])
        x = Activation("relu", name=name + "_out")(x)
        return x

    return apply


def ResidualStack(filters, blocks, stride=2, name=None):
    """A set of stacked residual blocks.
    Args:
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """

    def apply(x):
        x = ResidualBlock(filters, stride=stride, name=name + "_block1")(x)
        for i in range(2, blocks + 1):
            x = ResidualBlock(filters, conv_shortcut=False, name=name + "_block" + str(i))(x)

        return x

    return apply
