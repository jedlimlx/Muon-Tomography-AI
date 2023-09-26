import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers.SIREN import SIREN
from layers.convnext_block import ConvNeXtBlock
from layers.residual_block import ResidualBlock

_3d_base_params = {
    'widths': [256, 256, 256, 256, 256],
    'resolution': 64,
}


class PoolingModel(Model):
    def __init__(
            self,
            projection_dim=128,
            resolution=None,
            widths=(256, 256, 256, 256, 256),
            threshold=1e-8,
            *args, **kwargs
    ):
        super(PoolingModel, self).__init__(*args, **kwargs)

        self.projection_dim = projection_dim
        self.resolution = resolution
        self.widths = widths
        self.threshold = threshold

        # pooling the set of muon detections into a fixed-size vector
        self.projection = Sequential(
            [
                Dense(4 * projection_dim, activation="gelu"),
                Dense(projection_dim)
            ]
        )

        self.pooling = GlobalAveragePooling1D()

        self.latent_mlp = Sequential(
            [
                Dense(4 * projection_dim, activation="swish"),
                Dense(projection_dim)
            ]
        )

        # use siren to convert to 3D
        self.siren = SIREN(
            widths=widths,
            use_latent=True,
            output_dim=1
        )

        # coordinates to use with SIREN
        self.offsets = tf.repeat(
            tf.reshape(
                tf.stack(
                    tf.meshgrid(
                        tf.range(resolution, dtype=tf.float32),
                        tf.range(resolution, dtype=tf.float32),
                        tf.range(resolution, dtype=tf.float32),
                    ), axis=-1
                ) / resolution, (1, resolution**3, 3)
            ), 8, axis=0
        )

    def call(self, inputs, training=None, mask=None):
        # data format of inputs is x, y, z, px, py, pz, ver_x, ver_y, ver_z, ver_px, ver_py, ver_pz, p_estimate
        x = self.projection(inputs)
        x = self.pooling(x)
        latent = self.latent_mlp(x)

        latent = tf.reshape(latent, (-1, 1, self.projection_dim))

        # coordinates of the thing
        flattened = self.siren((self.offsets, latent))
        output = tf.reshape(flattened, (-1, 64, 64, 64))

        return output


if __name__ == "__main__":
    model = PoolingModel(
        **_3d_base_params
    )
    print(model(tf.random.normal((8, 20000, 13))).shape)

    model.summary()
