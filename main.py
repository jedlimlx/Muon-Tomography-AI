import random

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from tensorflow.keras.utils import plot_model

from layers import ResidualStack, ResidualBlock

# hyperparameters
ACTIVATION = "relu"
BLOCKS = [2, 2, 2, 2, 2]


# the skip connections in the U-Net
def skip_connection_2d_to_3d():
    def apply(x):
        x = Conv2D(x.shape[2], 3, activation=ACTIVATION, padding="same")(x)
        x = Reshape((x.shape[1], x.shape[2], x.shape[2], 1))(x)
        x = Conv3D(256, 3, activation=ACTIVATION, padding="same")(x)

        return x

    return apply


# a stack of blocks + pooling (returns the final block too)
def stack(filters, blocks, kernel_size=3, stride=2, name=None, activation="relu",
          drop_connect_rate=0.2, dims=2, downsample=True):
    def apply(x):
        conv = ResidualStack(filters, blocks, name=name, activation=activation,
                             drop_connect_rate=drop_connect_rate, kernel_size=kernel_size, dims=dims)(x)
        if downsample:
            # x = MaxPooling2D(2, name=name + "_pooling_block")(conv)
            x = ResidualBlock(filters, stride=stride, name=name + "_pooling_block", activation=activation,
                              drop_connect_rate=drop_connect_rate, kernel_size=kernel_size, dims=dims)(conv)
        else:
            x = UpSampling3D(size=(2, 2, 2), name=name + "_upsample")(conv)

        return x, conv

    return apply


def create_model(shape=(64, 64, 6)):
    output_2d = []
    output_3d = []
    output_skip = []

    inputs = Input(shape=shape)

    # Downward 2D part of U-Net
    for i in range(len(BLOCKS)):
        if i == 0:
            x = inputs

        x, conv = stack(64, BLOCKS[i], name=f"stack_2d_{i}")(x)
        output_2d.append(conv)
        output_skip.append(skip_connection_2d_to_3d()(conv))

    # Upward 3D part of U-Net
    for i in range(len(BLOCKS) - 1, -1, -1):
        if i == len(BLOCKS) - 1:
            x = output_skip[i]
        else:
            x = Concatenate()([output_skip[i], x])
            x = Conv3D(256, 3, activation=ACTIVATION, name=f"stack_3d_{i}_reshape", padding="same")(x)

        x, conv = stack(64, BLOCKS[i], dims=3, name=f"stack_3d_{i}", downsample=False)(x)
        output_3d.append(conv)

    model = Model(inputs=inputs, outputs=output_3d[-1])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    return model


def plot_voxels(data1):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(data1)
    plt.show()


def data_generator(directory=""):
    voxels_lst = []
    detections_lst = []
    for i in range(1000):
        detections = []
        for j in range(6):
            detections.append(np.expand_dims(np.load(f"{directory}/detections/{i}_orient_{j}.npy"), axis=-1))

        detections_lst.append(np.concatenate(detections, axis=-1))
        voxels_lst.append(1 / (np.expand_dims(np.load(f"{directory}/voxels/run_{i}.npy") + 1e-8, axis=-1)))

    return detections_lst, voxels_lst

