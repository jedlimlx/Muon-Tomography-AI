import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers import ResidualBlock, ConvNeXtBlock, DropBlock2D, DropBlock3D
from losses import binary_dice_coef_loss


# the skip connections in the U-Net
def skip_connection_2d_to_3d(filters=64, activation="relu", dims=3, name=None):
    def apply(x):
        if dims == 3:
            x = Conv2D(x.shape[2], 3, activation=activation, padding="same", name=name + "_conv2d")(x)
            x = Reshape((x.shape[1], x.shape[2], x.shape[2], 1), name=name + "_reshape")(x)
            x = Conv3D(filters, 3, activation=activation, padding="same", name=name + "_conv3d")(x)
        else:
            x = Conv2D(filters, 3, activation=activation, padding="same", name=name + "_conv2d")(x)

        return x

    return apply


# a stack of blocks + pooling (returns the final block too)
def stack(
        filters,
        blocks,
        kernel_size=3,
        stride=2,
        name=None,
        activation="relu",
        drop_connect_rate=0.2,
        dropout_rate=0.2,
        block_size=10,
        dims=2,
        downsample=True,
        downsample_filter=64,
        use_dropblock_2d=True,
        use_dropblock_3d=False,
        block_type="resnet"
):
    def apply(x):
        if block_type == "resnet":
            x = ResidualBlock(filters, name=name + "_block1", activation=activation,
                              drop_connect_rate=drop_connect_rate, kernel_size=kernel_size, dims=dims)(x)
            for i in range(2, blocks + 1):
                x = ResidualBlock(filters, conv_shortcut=False, name=name + "_block" + str(i), activation=activation,
                                  drop_connect_rate=drop_connect_rate, kernel_size=kernel_size, dims=dims)(x)
        elif block_type == "convnext":
            x = ConvNeXtBlock(filters, name=name + "_block1", activation=activation,
                              drop_connect_rate=drop_connect_rate, dims=dims)(x)
            for i in range(2, blocks + 1):
                x = ConvNeXtBlock(filters, name=name + "_block" + str(i), activation=activation,
                                  drop_connect_rate=drop_connect_rate, dims=dims)(x)

        conv = x

        if downsample:
            if use_dropblock_2d:
                x = DropBlock2D(keep_prob=1 - dropout_rate, block_size=block_size, name=name + "_dropblock2d")(x)
            else:
                x = Dropout(dropout_rate, name=name + "_dropout")(x)

            if block_type == "resnet":
                x = ResidualBlock(filters, stride=stride, name=name + "_pooling_block", activation=activation,
                                  drop_connect_rate=drop_connect_rate, kernel_size=kernel_size, dims=dims)(x)
            elif block_type == "convnext":
                downsample_layer = Sequential(
                    [
                        LayerNormalization(
                            epsilon=1e-6,
                            name=name + "_downsampling_layernorm",
                        ),
                        Conv2D(
                            downsample_filter,
                            kernel_size=2,
                            strides=2,
                            name=name + "_downsampling_conv",
                        ),
                    ],
                    name=name + "_downsampling_block",
                )

                x = downsample_layer(x)
        else:
            if dims == 3:
                if use_dropblock_3d:
                    x = DropBlock3D(keep_prob=1 - dropout_rate, block_size=block_size, name=name + "_dropblock3d")(x)
                else:
                    x = Dropout(dropout_rate, name=name + "_dropout")(x)

                x = UpSampling3D(size=(2, 2, 2), name=name + "_upsample")(x)
            else:
                if use_dropblock_2d:
                    x = DropBlock2D(keep_prob=1 - dropout_rate, block_size=block_size, name=name + "_dropblock2d")(x)
                else:
                    x = Dropout(dropout_rate, name=name + "_dropout")(x)

                x = UpSampling2D(size=(2, 2), name=name + "_upsample")(x)

        return x, conv

    return apply


def create_model(
        shape=(64, 64, 6),
        blocks=(2, 2, 2, 2, 2),
        filters=(32, 64, 128, 256, 512),
        activation="relu",
        drop_connect_rate=0.2,
        dropout_rate=0.2,
        block_size=10,
        noise=0.5,
        dropblock_2d=True,
        dropblock_3d=False,
        block_type="resnet",
        optimizer="adam",
        dimensions=2,
        loss=binary_dice_coef_loss(),
        weights=None
):
    output_2d = []
    output_3d = []
    output_skip = []

    inputs = Input(shape=shape)
    x = GaussianNoise(noise)(inputs)

    if block_type == "resnet":
        x = Conv2D(filters[0], 7, strides=1, use_bias=True, padding="same", name="stem")(x)
        x = BatchNormalization(name="stem_batch_norm")(x)
        x = Activation(activation, name="stem_activation")(x)
    elif block_type == "convnext":
        x = Conv2D(filters[0], 7, strides=1, use_bias=True, padding="same", name="stem")(x)
        x = LayerNormalization(name="stem_layer_norm")(x)
        x = Activation(activation, name="stem_activation")(x)

    # Downward 2D part of U-Net
    for i in range(len(blocks)):
        x, conv = stack(filters[i], blocks[i], name=f"stack_2d_{i}", drop_connect_rate=drop_connect_rate,
                        dropout_rate=dropout_rate, block_size=block_size, activation=activation,
                        use_dropblock_2d=dropblock_2d, use_dropblock_3d=dropblock_3d, block_type=block_type,
                        downsample_filter=filters[min(i + 1, len(filters) - 1)])(x)
        output_2d.append(conv)
        output_skip.append(skip_connection_2d_to_3d(filters[i], activation, name=f"skip_connection_{i}",
                                                    dims=dimensions)(conv))

    # Upward 2D / 3D part of U-Net
    for i in range(len(blocks) - 1, -1, -1):
        if i == len(blocks) - 1:
            x = output_skip[i]
        else:
            x = Concatenate()([output_skip[i], x])

            if dimensions == 3:
                x = Conv3D(filters[i], 3, activation=activation, name=f"stack_3d_{i}_reshape", padding="same")(x)
            else:
                x = Conv2D(filters[i], 3, activation=activation, name=f"stack_3d_{i}_reshape", padding="same")(x)

        x, conv = stack(filters[i], blocks[i], dims=dimensions, name=f"stack_3d_{i}", downsample=False,
                        drop_connect_rate=drop_connect_rate, dropout_rate=dropout_rate, block_size=block_size,
                        activation=activation, use_dropblock_2d=dropblock_2d, use_dropblock_3d=dropblock_3d,
                        block_type=block_type, downsample_filter=filters[min(i + 1, len(filters) - 1)])(x)
        output_3d.append(conv)

    if dimensions == 3:
        outputs = Conv3D(1, 3, padding="same", name="output")(output_3d[-1])
    else:
        outputs = Conv2D(1, 3, padding="same", name="output")(output_3d[-1])

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()

    if weights is not None: model.load_weights(weights)
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
