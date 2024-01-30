import tensorflow as tf

from keras.layers import *

from layers.regularisation import StochasticDepth


class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None, activation="relu",
                 drop_connect_rate=0.2, dims=2, **kwargs):
        """A residual block.
        Args:
          filters: integer, filters of the bottleneck layer.
          kernel_size: default 3, kernel size of the bottleneck layer.
          stride: default 1, stride of the first layer.
          conv_shortcut: default True, use convolution shortcut if True,
              otherwise identity shortcut.
          name: string, block label.
          activation: string, the activation function to use
          drop_connect_rate: string, the probability that the block will be skipped
          dims: integer, the number of dimensions of the convolution (2d or 3d)
        Returns:

          Output tensor for the residual block.
        """

        super().__init__(name=name, **kwargs)
        conv_block = Conv2D if dims == 2 else Conv3D

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_shortcut = conv_shortcut
        self.activation = activation
        self.drop_connect_rate = drop_connect_rate
        self.dims = dims

        self.shortcut = conv_block(filters, 1, strides=stride, use_bias=False, name=name + "_0_conv")
        self.shortcut_batch = BatchNormalization(name=name + "_0_bn")

        self.conv1 = conv_block(filters // 4, 1, strides=stride, use_bias=False, name=name + "_1_conv")
        self.batch_1 = BatchNormalization(name=name + "_1_bn")
        self.activation_1 = Activation(activation, name=name + f"_1_{activation}")

        self.conv2 = conv_block(filters // 4, kernel_size, padding="SAME", use_bias=False, name=name + "_2_conv")
        self.batch_2 = BatchNormalization(name=name + "_2_bn")
        self.activation_2 = Activation(activation, name=name + f"_2_{activation}")

        self.conv3 = conv_block(filters, 1, use_bias=False, name=name + "_3_conv")
        self.batch_3 = BatchNormalization(name=name + "_3_bn")

        self.sd = StochasticDepth(rate=drop_connect_rate, name=name + "_add")
        self.out = Activation(activation, name=name + "_out")

    def call(self, inputs, *args, **kwargs):
        x = inputs

        if self.conv_shortcut:
            shortcut = self.shortcut(x)
            shortcut = self.shortcut_batch(shortcut)
        else:
            shortcut = x

        x = self.conv1(x)
        x = self.batch_1(x)
        x = self.activation_1(x)

        x = self.conv2(x)
        x = self.batch_2(x)
        x = self.activation_2(x)

        x = self.conv3(x)
        x = self.batch_3(x)

        x = self.sd([shortcut, x])
        x = self.out(x)

        return x


def ResidualStack(filters, blocks, name=None, **kwargs):
    """A set of stacked residual blocks.
    Args:
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      name: string, stack label.
      kwargs: kwargs to pass into ResidualBlock
    Returns:
      Output tensor for the stacked blocks.
    """

    kwargs.pop("name", None)
    kwargs.pop("conv_shortcut", None)

    def apply(x):
        for i in range(1, blocks + 1):
            x = ResidualBlock(filters, conv_shortcut=False, name=name + "_block" + str(i), **kwargs)(x)

        return x

    return apply
