import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers import ResidualStack, ResidualBlock, DropBlock2D, DropBlock3D
from losses import binary_dice_coef_loss


# the skip connections in the U-Net
def skip_connection_2d_to_3d(filters=64, activation="relu"):
    def apply(x):
        x = Conv2D(x.shape[2], 3, activation=activation, padding="same")(x)
        x = Reshape((x.shape[1], x.shape[2], x.shape[2], 1))(x)
        x = Conv3D(4 * filters, 3, activation=activation, padding="same")(x)

        return x

    return apply


# a stack of blocks + pooling (returns the final block too)
def stack(filters, blocks, kernel_size=3, stride=2, name=None, activation="relu",
          drop_connect_rate=0.2, dropout_rate=0.2, block_size=10, dims=2, downsample=True,
          use_dropblock_2d=True, use_dropblock_3d=False):
    def apply(x):
        conv = ResidualStack(filters, blocks, name=name, activation=activation,
                             drop_connect_rate=drop_connect_rate, kernel_size=kernel_size, dims=dims)(x)
        if downsample:
            if use_dropblock_2d:
                x = DropBlock2D(keep_prob=1 - dropout_rate, block_size=block_size, name=name + "_dropblock2d")(conv)
            else:
                x = Dropout(dropout_rate, name=name + "_dropout")(x)

            x = ResidualBlock(filters, stride=stride, name=name + "_pooling_block", activation=activation,
                              drop_connect_rate=drop_connect_rate, kernel_size=kernel_size, dims=dims)(x)
        else:
            if use_dropblock_3d:
                x = DropBlock3D(keep_prob=1-dropout_rate, block_size=block_size, name=name + "_dropblock3d")(conv)
            else:
                x = Dropout(dropout_rate, name=name + "_dropout")(x)

            x = UpSampling3D(size=(2, 2, 2), name=name + "_upsample")(x)

        return x, conv

    return apply


def create_model(shape=(64, 64, 6), blocks=(2, 2, 2, 2, 2), filters=64, activation="relu",
                 drop_connect_rate=0.2, dropout_rate=0.2, block_size=10, optimizer="adam",
                 loss=binary_dice_coef_loss(), dropblock_2d=True, dropblock_3d=False, weights=None):
    output_2d = []
    output_3d = []
    output_skip = []

    inputs = Input(shape=shape)
    x = Conv2D(4 * filters, 3, padding="same", activation=activation)(inputs)

    # Downward 2D part of U-Net
    for i in range(len(blocks)):
        x, conv = stack(filters, blocks[i], name=f"stack_2d_{i}", drop_connect_rate=drop_connect_rate,
                        dropout_rate=dropout_rate, block_size=block_size,
                        activation=activation, use_dropblock_2d=dropblock_2d, use_dropblock_3d=dropblock_3d)(x)
        output_2d.append(conv)
        output_skip.append(skip_connection_2d_to_3d(filters, activation)(conv))

    # Upward 3D part of U-Net
    for i in range(len(blocks) - 1, -1, -1):
        if i == len(blocks) - 1:
            x = output_skip[i]
        else:
            x = Concatenate()([output_skip[i], x])
            x = Conv3D(4 * filters, 3, activation=activation, name=f"stack_3d_{i}_reshape", padding="same")(x)

        x, conv = stack(filters, blocks[i], dims=3, name=f"stack_3d_{i}", downsample=False,
                        drop_connect_rate=drop_connect_rate, dropout_rate=dropout_rate, block_size=block_size,
                        activation=activation, use_dropblock_2d=dropblock_2d, use_dropblock_3d=dropblock_3d)(x)
        output_3d.append(conv)

    outputs = Conv3D(1, 3, activation="sigmoid", padding="same", name="output")(output_3d[-1])

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()

    if weights is not None: model.load(weights)
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
        voxels_lst.append(np.expand_dims(np.load(f"{directory}/voxels/run_{i}.npy"), axis=-1))

    return detections_lst, voxels_lst


if __name__ == "__main__":
    model = create_model()
