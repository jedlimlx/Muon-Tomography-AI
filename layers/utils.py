from tensorflow.keras.layers import *
import inspect
import sys


def skip_connection_2d_to_3d(filters, activation="relu", name=None):
    """
    The skip connections in the U-Net
    Args:
        filters: Number of output filters
        activation: default 'relu', activation to use for Conv layers
        name: string, name of block

    Returns: Function that applies the skip connection layer.

    """
    def apply(x):
        x = Conv2D(x.shape[2], 3, activation=activation, padding="same", name=name + "_conv2d")(x)
        x = Reshape((x.shape[1], x.shape[2], x.shape[2], 1), name=name + "_reshape")(x)
        x = Conv3D(filters, 3, activation=activation, padding="same", name=name + "_conv3d")(x)

        return x

    return apply


def get_custom_objects():
    clsmembers = inspect.getmembers(sys.modules['layers'], inspect.isclass)
    clsmembers += inspect.getmembers(sys.modules['metrics'], inspect.isclass)
    return clsmembers


if __name__ == "__main__":
    print(get_custom_objects())
