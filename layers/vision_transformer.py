import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from functools import partial


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def compute_causal_mask(size):
    """Computes a causal mask (e.g., for masked self-attention layers).
    For example, if query and value both contain sequences of length 4,
    this function returns a boolean `Tensor` equal to:
    ```
    [[[True,  False, False, False],
      [True,  True,  False, False],
      [True,  True,  True,  False],
      [True,  True,  True,  True]]]
    ```
    Args:
      query: query `Tensor` of shape `(B, T, ...)`.
      value: value `Tensor` of shape `(B, S, ...)` (optional, defaults to
      query).
    Returns:
      mask: a boolean `Tensor` of shape [1, T, S] containing a lower
            triangular matrix of shape [T, S].
    """
    return tf.linalg.band_part(  # creates a lower triangular matrix
        tf.ones((1, size, size), tf.bool), -1, 0
    )


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MLP(Layer):
    def __init__(self, hidden_dim, out_dim, dropout_rate=0.1,
                 hidden_activation='gelu', out_activation='linear', name=None):
        """
        Represents the MLP used in vision transformers (2 dense layers)
        Args:
            hidden_dim: output dimensions of hidden layer
            out_dim: output dimensions of output layer
            dropout_rate: dropout rate (default 0.1)
            hidden_activation: activation for hidden layer (default 'gelu')
            out_activation: activation for output layer (default 'linear')
            name: name of the layer
        """
        super(MLP, self).__init__(name=name)
        self.dense1 = Dense(hidden_dim, activation=hidden_activation, name=f"{name}_dense_0")
        self.drop1 = Dropout(dropout_rate, name=f"{name}_dropout_0")
        self.dense2 = Dense(out_dim, out_activation, name=f"{name}_dense_1")
        self.drop2 = Dropout(dropout_rate, name=f"{name}_dropout_1")

    def call(self, inputs, *args, **kwargs):
        x = self.dense1(inputs)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        return x


# def MLP(hidden_units, dropout_rate, activation='gelu', name=None):
#     model = Sequential(name=name)
#
#     i = 0
#     for units in hidden_units:
#         model.add(Dense(units, activation=activation, name=name + f"_dense_{i}"))
#         model.add(Dropout(dropout_rate, name=name + f"_dropout_{i}"))
#         i += 1
#
#     return model


class EncoderBlock(Layer):
    def __init__(self, num_heads=16, dim=256, mlp_units=512, dropout=0.1, activation='gelu', name='vit_block',
                 norm=partial(LayerNormalization, epsilon=1e-5), **kwargs):
        super().__init__(name=name, **kwargs)
        self.norm1 = norm(name=f"{name}_norm_1")
        self.mhsa = MultiHeadAttention(num_heads=num_heads, key_dim=dim, dropout=dropout, name=f"{name}_mha")
        self.norm2 = norm(name=f"{name}_norm_2")
        self.mlp = MLP(hidden_dim=mlp_units, out_dim=dim, hidden_activation=activation, name=f"{name}_mlp")

    def call(self, inputs, **kwargs):
        x1 = self.norm1(inputs)
        attn = self.mhsa(x1, x1)
        x2 = attn + x1
        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        return x2 + x3


class DecoderBlock(Layer):
    def __init__(self, num_heads=16, enc_dim=256, dim=256, mlp_units=512, num_patches=256, dropout=0.1,
                 activation='gelu', name='decoder_block', norm=partial(LayerNormalization, epsilon=1e-5), **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_patches = num_patches
        self.norm1 = norm(name=f"{name}_norm_1")
        self.self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=dim, dropout=dropout,
                                                 name=f"{name}_self_attention")
        self.norm2 = norm(name=f"{name}_norm_2")
        self.cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=enc_dim, value_dim=dim, dropout=dropout,
                                                  name=f"{name}_cross_attention")
        self.norm3 = norm(name=f"{name}_norm_3")
        self.mlp = MLP(hidden_dim=mlp_units, out_dim=dim, hidden_activation=activation, name=f"{name}_mlp")

    def call(self, inputs, **kwargs):
        x, k = inputs
        x1 = self.norm1(x)
        x2 = self.self_attention(x1, x1, attention_mask=compute_causal_mask(self.num_patches))
        x2 = x2 + x1
        x2 = self.norm2(x2)
        x3 = self.cross_attention(x2, k)
        x3 = x3 + x2
        x3 = self.norm3(x3)
        x4 = self.mlp(x3)
        return x4 + x3


class Patches(Layer):
    def __init__(self, patch_width, patch_height, name=None):
        super(Patches, self).__init__(name=name)
        self.patch_width = patch_width
        self.patch_height = patch_height

    def call(self, images, **kwargs):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_width, self.patch_height, 1],
            strides=[1, self.patch_width, self.patch_height, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, name=None):
        super(PatchEncoder, self).__init__(name=name)
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch, **kwargs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class PatchDecoder(Layer):
    def __init__(self, patch_width, patch_height, x_patches, y_patches, name=None):
        super(PatchDecoder, self).__init__(name=name)
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.x_patches = x_patches
        self.y_patches = y_patches

        # self.positional_embedding = Embedding(
        #     input_dim=self.x_patches * self.y_patches, output_dim=projection_dim
        # )

        # self.positional_embedding = positional_encoding(256, projection_dim)

        # self.mlp = MLP(hidden_dim=mlp_units, out_dim=patch_width * patch_height, dropout_rate=0.1, name=f"{name}_mlp")
        # self.reshape = Reshape(target_shape=[self.patch_width, self.patch_height, 1],
        #                        input_shape=[1, self.patch_width * self.patch_height],
        #                        name=f"{name}_reshape_1")
        # self.concatenate = Concatenate(axis=-1)

    def call(self, encoded, **kwargs):
        # positions = tf.range(start=0, limit=self.x_patches * self.y_patches, delta=1)
        # embedding = self.positional_embedding(positions)

        # Extracting patches
        # patches = self.mlp(encoded + self.positional_embedding)
        reshaped = tf.reshape(encoded, (-1, self.y_patches, self.x_patches, self.patch_height, self.patch_width))
        reshaped = tf.transpose(reshaped, [0, 1, 3, 2, 4])
        reshaped = tf.reshape(reshaped, (-1, 256, 256, 1))

        # Merging into final output
        # def func(x):
        #     x = tf.nn.space_to_depth(x, self.patch_width)
        #     x = tf.reshape(x, [self.x_patches * self.patch_width, self.y_patches * self.patch_height, 1])
        #     return x

        return reshaped


def CTransformer(input_shape=(256, 256, 1), sinogram_height=1, sinogram_width=256, enc_dim=256, enc_layers=8,
                 enc_mlp_units=512, enc_heads=16, dec_projection=True, dec_dim=256, dec_layers=8, dec_heads=16,
                 dec_mlp_units=512, output_projection=True, output_patch_height=16, output_patch_width=16,
                 output_x_patches=16, output_y_patches=16, dropout=0.1,
                 activation='gelu', norm=partial(LayerNormalization, epsilon=1e-5)):
    # error checking
    if input_shape[0] % sinogram_height != 0 or input_shape[1] % sinogram_width != 0:
        raise ValueError("Cannot divide image into even patches")

    if (not dec_projection) and dec_dim != enc_dim:
        raise ValueError("Encoder and decoder dims must match")

    num_patches = int(input_shape[1] / sinogram_width * input_shape[0] / sinogram_height)

    if (not output_projection) and dec_dim != output_patch_height * output_patch_width:
        raise ValueError("Output patch size must patch decoder dims")

    inputs = Input(shape=input_shape)

    # patch
    x = Patches(patch_height=sinogram_height, patch_width=sinogram_width, name="patchify")(inputs)
    x = PatchEncoder(num_patches=num_patches, projection_dim=enc_dim, name="enc_projection")(x)

    # encoder
    for i in range(enc_layers):
        x = EncoderBlock(num_heads=enc_heads, mlp_units=enc_mlp_units, dim=enc_dim, dropout=dropout,
                         activation=activation, norm=norm, name=f"enc_block_{i}")(x)

    # decoder projection
    if dec_projection:
        x = PatchEncoder(num_patches=num_patches, projection_dim=dec_dim, name="dec_projection")(x)

    # decoder
    for i in range(dec_layers):
        x = EncoderBlock(num_heads=dec_heads, mlp_units=dec_mlp_units, dim=dec_dim, dropout=dropout,
                         activation=activation, norm=norm, name=f"dec_block_{i}")(x)

    # output projection
    if output_projection:
        x = PatchEncoder(num_patches=num_patches, projection_dim=output_patch_width * output_patch_height,
                         name="output_projection")(x)

    # reshape
    x = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches, name="depatchify")(x)

    return Model(inputs=inputs, outputs=x)


def CTranslator(input_shape=(256, 256, 1), sinogram_height=1, sinogram_width=256, enc_dim=256, enc_layers=8,
                enc_mlp_units=512, enc_heads=16, dec_dim=256, dec_layers=8, dec_heads=16,
                dec_mlp_units=512, output_projection=True, output_patch_height=16, output_patch_width=16,
                output_x_patches=16, output_y_patches=16, dropout=0.1,
                activation='gelu', norm=partial(LayerNormalization, epsilon=1e-5)):
    # error checking
    if input_shape[0] % sinogram_height != 0 or input_shape[1] % sinogram_width != 0:
        raise ValueError("Cannot divide image into even patches")

    num_patches = int(input_shape[1] / sinogram_width * input_shape[0] / sinogram_height)

    if (not output_projection) and dec_dim != output_patch_height * output_patch_width:
        raise ValueError("Output patch size must patch decoder dims")

    inputs = Input(shape=input_shape)
    targets = Input(shape=(output_y_patches * output_patch_height, output_x_patches * output_patch_width, 1))

    # encoder projection
    x = Patches(patch_height=sinogram_height, patch_width=sinogram_width, name="enc_patchify")(inputs)
    x = PatchEncoder(num_patches=num_patches, projection_dim=enc_dim, name="enc_projection")(x)

    # encoder
    for i in range(enc_layers):
        x = EncoderBlock(num_heads=enc_heads, mlp_units=enc_mlp_units, dim=enc_dim, dropout=dropout,
                         activation=activation, norm=norm, name=f"enc_block_{i}")(x)

    # decoder projection
    y = Patches(patch_height=output_patch_height, patch_width=output_patch_width, name="dec_patchify")(targets)
    y = PatchEncoder(num_patches=num_patches, projection_dim=dec_dim, name="dec_projection")(y)

    # decoder
    for i in range(dec_layers):
        y = DecoderBlock(num_heads=dec_heads, mlp_units=dec_mlp_units, dim=dec_dim, enc_dim=enc_dim,
                         num_patches=num_patches, dropout=dropout, activation=activation, norm=norm,
                         name=f"dec_block_{i}")([y, x])

    # output projection
    if output_projection:
        y = MLP(hidden_dim=dec_mlp_units, out_dim=output_patch_height * output_patch_width,
                hidden_activation=activation, name=f"output_projection")(y)

    # reshape
    y = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches, name="depatchify")(y)

    return Model(inputs=[inputs, targets], outputs=y)
