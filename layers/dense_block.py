import tensorflow as tf
keras = tf.keras

from keras import backend
from keras.layers import *
from keras.models import *


def DenseBlock(blocks, name=None, activation="relu"):
    """A dense block.
    Args:
      blocks: integer, the number of building blocks.
      name: string, block label.
      activation: string, the activation function to be used.
    Returns:
      a function that takes an input Tensor representing a DenseBlock.
    """

    def apply(x):
        for i in range(blocks):
            x = ConvBlock(32, name=f"{name}_block_{i}", activation=activation)(x)
        return x

    return apply


def TransitionBlock(reduction, name=None, activation="relu"):
    """A transition block.
    Args:
      reduction: float, compression rate at transition layers.
      name: string, block label.
      activation: string, the activation function to be used.
    Returns:
      a function that takes an input Tensor representing a TransitionBlock.
    """

    def apply(x):
        x = BatchNormalization(name=f"{name}_bn")(x)
        x = Activation(activation, name=f"{name}_{activation}")(x)
        x = Conv2D(int(backend.int_shape(x)[-1] * reduction), 1, use_bias=False, name=f"{name}_conv", )(x)
        x = AveragePooling2D(2, strides=2, name=f"{name}_pool")(x)
        return x

    return apply


def ConvBlock(growth_rate, name=None, activation="relu"):
    """A building block for a dense block.
    Args:
      growth_rate: float, growth rate at dense layers.
      name: string, block label.
      activation: string, the activation function to be used.
    Returns:
      a function that takes an input Tensor representing a ConvBlock.
    """

    def apply(x):
        x1 = BatchNormalization(name=f"{name}_0_bn")(x)
        x1 = Activation(activation, name=f"{name}_0_{activation}")(x1)
        x1 = Conv2D(4 * growth_rate, 1, use_bias=False, name=f"{name}_1_conv")(x1)
        x1 = BatchNormalization(name=f"{name}_1_bn")(x1)
        x1 = Activation(activation, name=f"{name}_1_{activation}")(x1)
        x1 = Conv2D(growth_rate, 3, padding="same", use_bias=False, name=f"{name}_2_conv")(x1)
        x = Concatenate(name=f"{name}_concat")([x, x1])
        return x

    return apply
