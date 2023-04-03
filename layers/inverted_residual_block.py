import tensorflow as tf
keras = tf.keras

from keras.layers import *


CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1.0 / 3.0,
        "mode": "fan_out",
        "distribution": "uniform",
    },
}

BN_AXIS = 3


def MBConvBlock(
    input_filters,
    output_filters,
    expand_ratio=4,
    kernel_size=3,
    strides=1,
    se_ratio=0.25,
    bn_momentum=0.9,
    activation="swish",
    survival_probability=0.8,
    dims=2,
    name=None,
):
    """MBConv block: Mobile Inverted Residual Bottleneck."""

    conv = Conv2D if dims == 2 else Conv3D
    dw_conv = DepthwiseConv2D if dims == 2 else Conv3D
    global_avg = GlobalAveragePooling2D if dims == 2 else GlobalAveragePooling3D

    def apply(inputs):
        # Expansion phase
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = conv(
                filters=filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                data_format="channels_last",
                use_bias=False,
                name=name + "expand_conv",
            )(inputs)
            x = BatchNormalization(
                momentum=bn_momentum,
                name=name + "expand_bn",
            )(x)
            x = Activation(activation, name=name + "expand_activation")(x)
        else:
            x = inputs

        # Depthwise conv
        x = dw_conv(
            kernel_size=kernel_size,
            strides=strides,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=name + "dwconv2",
        )(x)
        x = BatchNormalization(momentum=bn_momentum, name=name + "bn")(x)
        x = Activation(activation, name=name + "activation")(x)

        # Squeeze and excite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = global_avg(name=name + "se_squeeze")(x)

            se_shape = (1, 1, 1, filters) if dims == 3 else (1, 1, filters)
            se = Reshape(se_shape, name=name + "se_reshape")(se)
            se = conv(
                filters_se,
                1,
                padding="same",
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )(se)
            se = conv(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )(se)

            x = multiply([x, se], name=name + "se_excite")

            # Output phase
            x = conv(
                filters=output_filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                data_format="channels_last",
                use_bias=False,
                name=name + "project_conv",
            )(x)
            x = BatchNormalization(
                axis=BN_AXIS, momentum=bn_momentum, name=name + "project_bn"
            )(x)

            if strides == 1 and input_filters == output_filters:
                if survival_probability:
                    x = Dropout(
                        survival_probability,
                        noise_shape=(None, 1, 1, 1),
                        name=name + "drop",
                    )(x)
                x = add([x, inputs], name=name + "add")
        return x

    return apply


def FusedMBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability: float = 0.8,
    name=None,
):
    """Fused MBConv Block: Fusing the proj conv1x1 and depthwise_conv into a
    conv2d."""

    def apply(inputs):
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                data_format="channels_last",
                padding="same",
                use_bias=False,
                name=name + "expand_conv",
            )(inputs)
            x = BatchNormalization(
                axis=BN_AXIS, momentum=bn_momentum, name=name + "expand_bn"
            )(x)
            x = Activation(
                activation=activation, name=name + "expand_activation"
            )(x)
        else:
            x = inputs

        # Squeeze and excite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = GlobalAveragePooling2D(name=name + "se_squeeze")(x)
            if BN_AXIS == 1:
                se_shape = (filters, 1, 1)
            else:
                se_shape = (1, 1, filters)

            se = Reshape(se_shape, name=name + "se_reshape")(se)

            se = Conv2D(
                filters_se,
                1,
                padding="same",
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )(se)
            se = Conv2D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )(se)

            x = multiply([x, se], name=name + "se_excite")

        # Output phase:
        x = Conv2D(
            output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1 if expand_ratio != 1 else strides,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name=name + "project_conv",
        )(x)
        x = BatchNormalization(
            axis=BN_AXIS, momentum=bn_momentum, name=name + "project_bn"
        )(x)
        if expand_ratio == 1:
            x = Activation(
                activation=activation, name=name + "project_activation"
            )(x)

        # Residual:
        if strides == 1 and input_filters == output_filters:
            if survival_probability:
                x = Dropout(
                    survival_probability,
                    noise_shape=(None, 1, 1, 1),
                    name=name + "drop",
                )(x)
            x = add([x, inputs], name=name + "add")
        return x

    return apply
