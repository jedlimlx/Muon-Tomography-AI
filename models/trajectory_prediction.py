import tensorflow as tf
keras = tf.keras

from keras.models import Model, Sequential
from keras.layers import *

from keras_nlp.layers import SinePositionEncoding


from layers import ConvNeXtBlock


class TransformerDecoder(Layer):  # todo use keras cv implementation when it becomes better
    """
    Transformer decoder block implementation as a Keras Layer.

    Args:
        project_dim: the dimensionality of the projection of the encoder, and
            output of the `MultiHeadAttention`
        mlp_dim: the intermediate dimensionality of the MLP head before
            projecting to `project_dim`
        num_heads: the number of heads for the `MultiHeadAttention` layer
        mlp_dropout: default 0.1, the dropout rate to apply between the layers
            of the MLP head of the encoder
        attention_dropout: default 0.1, the dropout rate to apply in the
            MultiHeadAttention layer
        activation: default 'tf.activations.gelu', the activation function to
            apply in the MLP head - should be a function
        layer_norm_epsilon: default 1e-06, the epsilon for `LayerNormalization`
            layers
    """

    def __init__(
            self,
            project_dim,
            num_heads,
            mlp_dim,
            mlp_dropout=0.1,
            attention_dropout=0.1,
            activation=keras.activations.gelu,
            layer_norm_epsilon=1e-06,
            divide_heads=True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = Activation(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.mlp_units = [mlp_dim, project_dim]

        self.divide_heads = divide_heads

        # layer norms
        self.layer_norm1 = LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.layer_norm2 = LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.layer_norm3 = LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )

        # attention
        self.attn = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.project_dim // (self.num_heads if self.divide_heads else 1),
            dropout=self.attention_dropout
        )

        # MLP projection
        self.dense1 = Dense(self.mlp_units[0])
        self.dense2 = Dense(self.mlp_units[1])

        self.dropout = Dropout(self.mlp_dropout)

    def build(self, input_shape):
        self.cross_attn = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=input_shape[1][-1] // (self.num_heads if self.divide_heads else 1),
            dropout=self.attention_dropout
        )

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs

        if encoder_inputs.shape[-1] != self.project_dim:
            raise ValueError(
                "The input and output dimensionality must be the same, but the "
                f"TransformerDecoder was provided with {inputs.shape[-1]} and "
                f"{self.project_dim}"
            )

        # self-attention part
        x = self.layer_norm1(encoder_inputs)
        x = self.attn(x, x)
        x = self.dropout(x)
        x = x + encoder_inputs

        y = self.layer_norm2(x)
        y = self.attn(y, decoder_inputs)
        y = self.dropout(y)
        y = x + y

        z = self.layer_norm3(y)
        z = self.dense1(z)
        if self.activation == keras.activations.gelu:
            z = self.activation(z, approximate=True)
        else:
            z = self.activation(z)

        z = self.dropout(z)
        z = self.dense2(z)
        z = self.dropout(z)

        output = y + z
        return output

    def get_config(self):
        config = super().get_config()
        activation = self.activation
        if not isinstance(activation, (str, dict)):
            activation = keras.activations.serialize(activation)
        config.update(
            {
                "project_dim": self.project_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
                "mlp_dropout": self.mlp_dropout,
                "activation": activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        activation = config.pop("activation")
        if isinstance(activation, (str, dict)):
            activation = keras.activations.deserialize(activation)
        return cls(activation=activation, **config)


class TrajectoryPrediction(Model):
    def __init__(
        self,
        projection_dim=64,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.input_projection = Dense(64)
        self.positional_embedding = SinePositionEncoding()
        self.positional_embedding_2 = SinePositionEncoding()

        self.output_projection = Dense(2)

        self.encoder = Sequential(
            [
                Conv3D(32, 7, activation="gelu", padding="same"),
                Conv3D(64, 7, activation="gelu", padding="same"),
                MaxPooling3D(2),
                ConvNeXtBlock(64, dims=3),
                ConvNeXtBlock(64, dims=3),
                ConvNeXtBlock(64, dims=3),
                MaxPooling3D(2),
                ConvNeXtBlock(64, dims=3),
                ConvNeXtBlock(64, dims=3),
                ConvNeXtBlock(64, dims=3),
                MaxPooling3D(2),
                ConvNeXtBlock(64, dims=3),
                ConvNeXtBlock(64, dims=3),
                ConvNeXtBlock(64, dims=3),
                MaxPooling3D(2)
            ]
        )

        self.decoder = [
            TransformerDecoder(
                project_dim=projection_dim,
                num_heads=16,
                mlp_dim=256,
                mlp_dropout=0,
                attention_dropout=0
            ) for _ in range(4)
        ]

    def call(self, inputs, training=None, mask=None):
        density, x = inputs

        # run density through some ConvNeXt blocks
        latent = self.encoder(density)
        latent = tf.reshape(latent, (-1, 64, 64))
        latent = latent + self.positional_embedding_2(latent)

        # expand and add positional embedding
        x = self.input_projection(x)
        x = x + self.positional_embedding(x)

        # run transformer decoder block between x and the latent
        for decoder_layer in self.decoder:
            x = decoder_layer((x, latent))

        # output projection
        x = self.output_projection(x)

        return x


if __name__ == "__main__":
    from data_generation import predict_trajectory

    # List of materials by radiation length
    radiation_lengths = [
        1000000000000000000000000,  # air
        49.834983498349835,  # benzene
        49.82309830679809,  # methanol
        36.08,  # water
        14.385057471264368,  # magnesium
        11.552173913043479,  # concrete
        10.607758620689655,  # gypsum
        10.412903225806451,  # calcium
        9.75,  # sulfur
        9.368827823100043,  # silicon
        8.895887365690998,  # aluminium
        4.436732514682328,  # caesium
        1.967741935483871,  # manganese
        1.7576835153670307,  # iron
        1.7200811359026373,  # iodine
        1.4243990114580993,  # nickel
        0.9589041095890413,  # molybdenum
        0.8542857142857143,  # silver
        0.6609442060085837,  # polonium
        0.5612334801762114,  # lead
        0.33436853002070394,  # gold
        0.316622691292876  # uranium
    ]

    inverse_radiation_length = [1/x for x in radiation_lengths]

    # Create a description of the features.
    feature_description = {
        'trajectory': tf.io.FixedLenFeature([], tf.string),
        'voxels': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_example(example_proto):
        res = tf.io.parse_single_example(example_proto, feature_description)
        y = tf.io.parse_tensor(res['trajectory'], out_type=tf.double)
        voxels = tf.io.parse_tensor(res['voxels'], out_type=tf.int32)
        voxels.set_shape((64, 64, 64))

        y = tf.cast(y, dtype=tf.float32)
        return tf.repeat(voxels[tf.newaxis, ...], tf.shape(y)[0], axis=0), y

    def process_data(voxels, trajectory):
        init = tf.concat(
            [
                predict_trajectory(
                    trajectory[0, 0], trajectory[-1, 0],
                    -(trajectory[1, 0] - trajectory[0, 0]) * 63,
                    -(trajectory[-1, 0] - trajectory[-2, 0]) * 63
                )[..., tf.newaxis],
                predict_trajectory(
                    trajectory[0, 1], trajectory[-1, 1],
                    -(trajectory[1, 1] - trajectory[0, 1]) * 63,
                    -(trajectory[-1, 1] - trajectory[-2, 1]) * 63
                )[..., tf.newaxis]
            ], axis=-1
        )
        tf.gather_nd(
            inverse_radiation_length,
            tf.cast(voxels[..., tf.newaxis], tf.int32)
        )
        return (voxels, init), trajectory

    root = r"D:\muons_data\muons_trajectory"
    ds = (
        tf.data.TFRecordDataset(f'{root}/trajectory_prediction.tfrecord', compression_type="GZIP")
        .map(_parse_example)
        .unbatch()
        .map(process_data)
        .batch(8)
    )

    # model = TrajectoryPrediction()
    # model((tf.random.normal((8, 64, 64, 64, 1)), tf.random.normal((8, 64, 2))))

    # model.summary()
