import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers.poca import poca
from layers.agg_3d import ScatterAndAvg3D
from layers.gaussian_agg3d import GaussianScatterAndAvg3D
from layers.convnext_block import ConvNeXtBlock
from layers.residual_block import ResidualBlock


_3d_base_params = {
    'point_size': 3,
    'downward_convs': [1, 1, 2, 3, 5],
    'downward_filters': [4, 16, 64, 128, 256],
    'upward_convs': [4, 3, 2, 1],
    'upward_filters': [128, 64, 16, 4],
    'resolution': 64,
}


class Agg3D(Model):
    def __init__(
            self,
            point_size=3,
            downward_convs=None,
            downward_filters=None,
            upward_convs=None,
            upward_filters=None,
            resolution=None,
            threshold=1e-8,
            poca_nn=None,
            hidden_layers=(8, 4),
            use_lstm=False,
            use_residual=False,
            use_gaussian=False,
            *args, **kwargs
    ):
        super(Agg3D, self).__init__(*args, **kwargs)
        self.resolution = resolution
        self.point_size = point_size
        self.downward_filters = downward_filters

        self.poca_nn = poca_nn
        self.use_lstm = use_lstm
        self.threshold = threshold

        self.hidden_layers = hidden_layers

        if not use_gaussian:
            self.agg = ScatterAndAvg3D(
                resolution=resolution,
                channels=downward_filters[0],
                point_size=point_size,
                projection_dim=point_size ** 3 * downward_filters[0],
                hidden_layers=hidden_layers,
                use_lstm=use_lstm
            )
        else:
            self.agg = GaussianScatterAndAvg3D(
                resolution=resolution,
                channels=downward_filters[0],
                point_size=point_size,
                projection_dim=downward_filters[0],
                hidden_layers=hidden_layers,
                use_lstm=use_lstm
            )

        if use_residual: conv = ResidualBlock
        else: conv = ConvNeXtBlock

        # downward ConvNeXt blocks
        self.downward_convs = []
        for stage in range(len(downward_convs)):
            stack = []
            for c in range(downward_convs[stage]):
                stack.append(conv(
                    downward_filters[stage],
                    kernel_size=3,
                    dims=3,
                    name=f'{self.name}/downward_stage_{stage}/block_{c}'
                ))
            self.downward_convs.append(Sequential(stack, name=f'{self.name}/downward_stage_{stage}'))

        self.downsampling = []
        self.upsampling = []
        self.upward_convs = []
        for stage in range(len(downward_convs) - 1):
            # downsampling convolutions
            self.downsampling.append(Sequential([
                LayerNormalization(epsilon=1e-6, name=f'{self.name}/downward_stage_{stage}/downsampling/layer_norm'),
                Conv3D(filters=downward_filters[stage + 1],
                       kernel_size=2,
                       strides=2,
                       name=f'{self.name}/downward_stage_{stage}/downsampling/conv2d',
                       padding='same')
            ], name=f'{self.name}/downward_stage_{stage}/downsampling'))

            # upsampling + convolutions
            self.upsampling.append(Sequential([
                Dense(units=upward_filters[stage], name=f'{self.name}/upward_stage_{stage}/upsampling/pointwise_conv'),
                LayerNormalization(epsilon=1e-6, name=f'{self.name}/upward_stage_{stage}/upsampling/layer_norm'),
                UpSampling3D(name=f'{self.name}/upward_stage_{stage}/upsampling/upsampling'),
            ]))

            # upward ConvNeXt blocks
            stack = []
            for c in range(upward_convs[stage]):
                stack.append(conv(
                    upward_filters[stage],
                    kernel_size=3,
                    dims=3,
                    name=f'{self.name}/upward_stage_{stage}/block_{c}'
                ))
            self.upward_convs.append(Sequential(stack, name=f'{self.name}/upward_stage_{stage}'))

        self.final_conv = Conv3D(1, 1, name=f'{self.name}/final_conv')

    def call(self, inputs, training=None, mask=None):
        # data format of inputs is x, y, z, px, py, pz, ver_x, ver_y, ver_z, ver_px, ver_py, ver_pz, p_estimate
        positions = poca(*tf.split(inputs[..., :12], 4, axis=-1), self.threshold, self.poca_nn)
        x = self.agg([positions, inputs])

        skip_outputs = []
        for i, block in enumerate(self.downward_convs):
            x = block(x)
            if i != len(self.downsampling):
                skip_outputs.append(x)
                x = self.downsampling[i](x)

        for i, block in enumerate(self.upward_convs):
            x = self.upsampling[i](x)
            x += skip_outputs.pop()
            x = block(x)

        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    model = Agg3D(
        **{
            'point_size': 3,
            'downward_convs': [1, 1, 2, 3, 5],
            'downward_filters': [8, 16, 64, 128, 256],
            'upward_convs': [4, 3, 2, 1],
            'upward_filters': [128, 64, 16, 8],
            'resolution': 64,
            'threshold': 1e-3,
            'hidden_layers': [8, 4],
            'use_residual': True,
            'use_lstm': False,
            # 'use_gaussian': False
        }  # , poca_nn=poca_nn
    )
    print(model(tf.random.normal((8, 20000, 13))).shape)

    model.summary()
