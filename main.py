import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers import ResidualBlock, ConvNeXtBlock, DropBlock2D, DropBlock3D, MBConvBlock
from layers import Patches, PatchEncoder, PatchDecoder
from losses import binary_dice_coef_loss

from metrics import get_flops


# the skip connections in the U-Net
def skip_connection_2d_to_3d(filters=64, activation="relu", init_dims=2, final_dims=3, name=None):
    def apply(x):
        if init_dims == 2 and final_dims == 3:
            x = Conv2D(x.shape[2], 3, activation=activation, padding="same", name=name + "_conv2d")(x)
            x = Reshape((x.shape[1], x.shape[2], x.shape[2], 1), name=name + "_reshape")(x)
            x = Conv3D(filters, 3, activation=activation, padding="same", name=name + "_conv3d")(x)
        elif init_dims == 2 and final_dims == 2:
            x = Conv2D(filters, 3, activation=activation, padding="same", name=name + "_conv2d")(x)
        elif init_dims == 1 and final_dims == 2:
            x = Conv1D(x.shape[1], 3, activation=activation, padding="same", name=name + "_conv1d")(x)
            x = Reshape((x.shape[1], x.shape[1], 1), name=name + "_reshape")(x)
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
        block_type="resnet",
        attention="se"
):
    def apply(x):
        if block_type == "resnet":
            x = ResidualBlock(filters, name=name + "_block1", activation=activation,
                              drop_connect_rate=drop_connect_rate, kernel_size=kernel_size, dims=dims,
                              attention=attention)(x)
            for i in range(2, blocks + 1):
                x = ResidualBlock(filters, conv_shortcut=False, name=name + "_block" + str(i), activation=activation,
                                  drop_connect_rate=drop_connect_rate, kernel_size=kernel_size, dims=dims,
                                  attention=attention)(x)
        elif block_type == "convnext":
            x = ConvNeXtBlock(filters, name=name + "_block1", activation=activation, kernel_size=kernel_size,
                              drop_connect_rate=drop_connect_rate, dims=dims, attention=attention)(x)
            for i in range(2, blocks + 1):
                x = ConvNeXtBlock(filters, name=name + "_block" + str(i), activation=activation, kernel_size=kernel_size,
                                  drop_connect_rate=drop_connect_rate, dims=dims, attention=attention)(x)
        elif block_type == "efficientnet":
            for i in range(1, blocks):
                x = MBConvBlock(filters, filters, name=name + "_block" + str(i), activation=activation,
                                kernel_size=kernel_size, dims=dims)(x)

            x = MBConvBlock(filters, downsample_filter, name=name + f"_block{blocks}", activation=activation,
                            kernel_size=kernel_size, dims=dims)(x)

        conv = x

        if downsample:
            if use_dropblock_2d and dims == 2:
                x = DropBlock2D(keep_prob=1 - dropout_rate, block_size=block_size, name=name + "_dropblock2d")(x)
            else:
                x = Dropout(dropout_rate, name=name + "_dropout")(x)

            if block_type == "resnet":
                x = ResidualBlock(filters, stride=stride, name=name + "_pooling_block", activation=activation,
                                  drop_connect_rate=drop_connect_rate, kernel_size=kernel_size, dims=dims)(x)
            elif block_type == "convnext":
                conv_1 = Conv2D if dims == 2 else Conv1D
                downsample_layer = Sequential(
                    [
                        LayerNormalization(
                            epsilon=1e-6,
                            name=name + "_downsampling_layernorm",
                        ),
                        conv_1(
                            downsample_filter,
                            kernel_size=2,
                            strides=2,
                            name=name + "_downsampling_conv",
                        ),
                    ],
                    name=name + "_downsampling_block",
                )

                x = downsample_layer(x)
            elif block_type == "efficientnet":
                x = Conv2D(
                    downsample_filter,
                    kernel_size=2,
                    strides=2,
                    name=name + "_downsampling_conv",
                )(x)
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


def round_filters(filters, divisor=8, width_coefficient=1):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient=1):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


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


def sam_train_step(self, data, rho=0.05, eps=1e-12):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    if len(data) == 3:
        x, y, sample_weight = data
    else:
        sample_weight = None
        x, y = data

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # first step
    e_ws = []
    grad_norm = tf.linalg.global_norm(gradients)
    ew_multiplier = rho / (grad_norm + eps)
    for i in range(len(trainable_vars)):
        e_w = tf.math.multiply(gradients[i], ew_multiplier)
        trainable_vars[i].assign_add(e_w)
        e_ws.append(e_w)

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    for i in range(len(trainable_vars)):
        trainable_vars[i].assign_sub(e_ws[i])
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Update the metrics.
    # Metrics are configured in `compile()`.
    self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

    # Return a dict mapping metric names to current value.
    # Note that it will include the loss (tracked in self.metrics).
    return {m.name: m.result() for m in self.metrics}


class TomographyModel(Model):
    def __init__(self, params, use_sam=True, **kwargs):
        super(TomographyModel, self, **kwargs).__init__()

        if params["task"] == "sparse":
            output_2d = []
            output_3d = []
            output_skip = []

            inputs = Input(shape=params["shape"])
            x = GaussianNoise(params["noise"])(inputs)

            conv_1 = Conv1D if params["initial_dimensions"] == 1 else Conv2D
            if params["block_type"] == "resnet" or params["block_type"] == "efficientnet":
                x = conv_1(params["filters"][0], params["kernel_size"], strides=1, use_bias=True, padding="same", name="stem")(x)
                x = BatchNormalization(name="stem_batch_norm")(x)
                x = Activation(params["activation"], name="stem_activation")(x)
            elif params["block_type"] == "convnext":
                x = conv_1(params["filters"][0], params["kernel_size"], strides=1, use_bias=True, padding="same", name="stem")(x)
                x = LayerNormalization(name="stem_layer_norm")(x)
                x = Activation(params["activation"], name="stem_activation")(x)

            # Downward 2D part of U-Net
            for i in range(len(params["blocks"])):
                x, conv = stack(
                    params["filters"][i],
                    params["blocks"][i],
                    name=f"stack_2d_{i}",
                    drop_connect_rate=params["drop_connect_rate"],
                    dropout_rate=params["dropout_rate"],
                    block_size=params["block_size"],
                    activation=params["activation"],
                    kernel_size=params["kernel_size"],
                    use_dropblock_2d=params["dropblock_2d"],
                    use_dropblock_3d=params["dropblock_3d"],
                    block_type=params["block_type"],
                    downsample_filter=params["filters"][min(i + 1, len(params["filters"]) - 1)],
                    attention=params["attention"],
                    dims=params["initial_dimensions"]
                )(x)
                output_2d.append(conv)
                output_skip.append(
                    skip_connection_2d_to_3d(
                        params["filters"][i],
                        params["activation"],
                        name=f"skip_connection_{i}",
                        init_dims=params["initial_dimensions"],
                        final_dims=params["dimensions"]
                    )(conv)
                )

            # Upward 2D / 3D part of U-Net
            for i in range(len(params["blocks"]) - 1, -1, -1):
                if i == len(params["blocks"]) - 1:
                    x = output_skip[i]
                else:
                    x = Concatenate()([output_skip[i], x])

                    if params["dimensions"] == 3:
                        x = Conv3D(
                            params["filters"][i], 3,
                            activation=params["activation"],
                            name=f"stack_3d_{i}_reshape",
                            padding="same"
                        )(x)
                    else:
                        x = Conv2D(
                            params["filters"][i], 3,
                            activation=params["activation"],
                            name=f"stack_3d_{i}_reshape",
                            padding="same"
                        )(x)

                x, conv = stack(
                    params["filters"][i],
                    params["blocks"][i],
                    dims=params["dimensions"],
                    name=f"stack_3d_{i}",
                    downsample=False,
                    drop_connect_rate=params["drop_connect_rate"],
                    dropout_rate=params["dropout_rate"],
                    block_size=params["block_size"],
                    activation=params["activation"],
                    kernel_size=params["kernel_size"],
                    use_dropblock_2d=params["dropblock_2d"],
                    use_dropblock_3d=params["dropblock_3d"],
                    block_type=params["block_type"],
                    downsample_filter=params["filters"][min(i + 1, len(params["filters"]) - 1)],
                    attention=params["attention"]
                )(x)
                output_3d.append(conv)

            if params["dimensions"] == 3:
                x = Conv3D(params["filters"][0], params["kernel_size"], padding="same", name="final",
                                 activation=params["activation"])(output_3d[-1])
                outputs = Conv3D(1, 3, padding="same", name="output", activation=params["final_activation"])(x)
            else:
                x = Conv2D(params["filters"][0], params["kernel_size"], padding="same", name="final",
                                 activation=params["activation"])(output_3d[-1])
                outputs = Conv2D(1, 3, padding="same", name="output", activation=params["final_activation"])(x)
        elif params["task"] == "ct":
            inputs = Input(shape=(params["sinogram_width"], params["num_sinograms"], 1))

            # Creating and encoding patches
            x = Patches(params["sinogram_width"], 1, name="extract_patches")(inputs)
            x = PatchEncoder(params["num_sinograms"], params["projection_dims"], name="encode_patches")(x)

            # Create multiple layers of the Transformer block.
            for i in range(len(params["num_heads"])):
                x = ViTBlock(
                    params["num_heads"][i],
                    params["projection_dims"],
                    params["transformer_units"][i],
                    name=f"block_{i}"
                )(x)

            # Creating the feature tensor
            # x = LayerNormalization(epsilon=1e-6, name="layer_norm")(x)
            # x = Flatten()(x)
            x = Dropout(params["dropout"], name="dropout")(x)
            outputs = PatchDecoder(
                params["patch_size"],
                params["patch_size"],
                int(params["num_patches"] ** 0.5),
                int(params["num_patches"] ** 0.5),
                params["projection_dims"],
                params["decoder_units"],
                name="patch_decoder"
            )(x)

        self.params = params
        self.use_sam = use_sam
        self.model = Model(inputs=inputs, outputs=outputs)

    def call(self, inputs):
        return self.model(inputs)

    def flops(self):
        return get_flops(model, [tf.zeros((1,) + self.params["shape"])])

    def train_step(self, data):
        if self.use_sam:
            return sam_train_step(self, data, rho=self.params["rho"])
        else:
            return super().train_step(data)

    def summary(self):
        self.model.summary()

    @staticmethod
    def _gradients_order2_norm(gradients):
        norm = tf.norm(tf.stack([tf.norm(grad) for grad in gradients if grad is not None]))
        return norm


def create_model(
        params,
        optimizer="adam",
        loss=binary_dice_coef_loss(),
        weights=None
):
    model = TomographyModel(params)
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()

    if weights is not None: model.load_weights(weights)
    return model


if __name__ == "__main__":
    model = create_model(
        {
            "task": "sparse",
            "shape": (64, 64, 6),
            "blocks": (1, 2, 2, 3, 4),
            "filters": (64, 64, 64, 64, 64),
            "activation": "swish",
            "kernel_size": 7,
            "drop_connect_rate": 0.05,
            "dropout_rate": 0.05,
            "block_size": 10,
            "noise": 0.20,
            "dropblock_2d": True,
            "dropblock_3d": False,
            "block_type": "convnext",
            "attention": "coatnet",
            "dimensions": 3,
            "initial_dimensions": 2,
            "final_activation": "sigmoid"
        }
    )

    print(f"GFLOPS: {model.flops()}")

