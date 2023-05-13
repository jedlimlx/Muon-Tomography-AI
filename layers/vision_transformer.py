import tensorflow as tf

from data_generation.lpradon import LPRadonForward

keras = tf.keras

import numpy as np

from keras.layers import *
from keras.models import *

from functools import partial

from utils import add_noise, preprocess_data


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
    def __init__(self, hidden_dim, out_dim, dropout_rate=0.,
                 hidden_activation='gelu', out_activation='linear', name=None, **kwargs):
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
        super(MLP, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
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

    def get_config(self):
        cfg = super(MLP, self).get_config()
        cfg.update({
            'hidden_dim': self.hidden_dim,
            'out_dim': self.out_dim,
            'dropout_rate': self.dropout_rate,
            'hidden_activation': self.hidden_activation,
            'out_activation': self.out_activation
        })
        return cfg


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
    def __init__(self, num_heads=16, dim=256, mlp_units=512, dropout=0., out_dim=None, activation='gelu',
                 name='vit_block', norm=partial(LayerNormalization, epsilon=1e-5), **kwargs):
        super().__init__(name=name, **kwargs)
        if out_dim is None:
            out_dim = dim

        self.num_heads = num_heads
        self.dim = dim
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.out_dim = out_dim
        self.activation = activation

        self.norm1 = norm(name=f"{name}_norm_1")
        self.mhsa = MultiHeadAttention(num_heads=num_heads, key_dim=dim, dropout=dropout, name=f"{name}_mha")
        self.norm2 = norm(name=f"{name}_norm_2")
        self.mlp = MLP(hidden_dim=mlp_units, out_dim=out_dim, hidden_activation=activation, name=f"{name}_mlp")

    def call(self, inputs, **kwargs):
        x1 = self.norm1(inputs)
        attn = self.mhsa(x1, x1)
        x2 = attn + x1
        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        return x2 + x3

    @classmethod
    def from_config(cls, config):
        norm_cls = deserialize(config['norm']).__class__
        del config['norm']['config']['name']
        norm = partial(norm_cls, **config['norm']['config'])
        del config['norm']

        return cls(**config, norm=norm)

    def get_config(self):
        cfg = super(EncoderBlock, self).get_config()
        cfg.update({
            'num_heads': self.num_heads,
            'dim': self.dim,
            'mlp_units': self.mlp_units,
            'dropout': self.dropout,
            'out_dim': self.out_dim,
            'activation': self.activation,
            'norm': serialize(self.norm1)
        })

        return cfg


class DecoderBlock(Layer):
    def __init__(self, num_heads=16, enc_dim=256, dim=256, mlp_units=512, num_patches=256, dropout=0.1,
                 activation='gelu', name='decoder_block', norm=partial(LayerNormalization, epsilon=1e-5), **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_heads = num_heads
        self.enc_dim = enc_dim
        self.dim = dim
        self.mlp_units = mlp_units
        self.num_patches = num_patches
        self.dropout = dropout
        self.activation = activation

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

    @classmethod
    def from_config(cls, config):
        norm_cls = deserialize(config['norm']).__class__
        del config['norm']['config']['name']
        norm = partial(norm_cls, **config['norm']['config'])
        del config['norm']

        return cls(**config, norm=norm)

    def get_config(self):
        cfg = super(DecoderBlock, self).get_config()
        cfg.update({
            'num_heads': self.num_heads,
            'enc_dim': self.enc_dim,
            'dim': self.dim,
            'mlp_units': self.mlp_units,
            'num_patches': self.num_patches,
            'dropout': self.dropout,
            'activation': self.activation,
            'norm': serialize(self.norm1)
        })

        return cfg


class Patches(Layer):
    def __init__(self, patch_width, patch_height, name=None, **kwargs):
        super(Patches, self).__init__(name=name, **kwargs)
        self.patch_width = patch_width
        self.patch_height = patch_height

    def call(self, images, **kwargs):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_height, self.patch_width, 1],
            strides=[1, self.patch_height, self.patch_width, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        cfg = super(Patches, self).get_config()
        cfg.update({
            'patch_width': self.patch_width,
            'patch_height': self.patch_height
        })

        return cfg


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, embedding_type='sin_cos', name=None, **kwargs):
        """
        Projection and embedding
        Args:
            num_patches: Sequence length
            projection_dim: Output dim after projection
            embedding_type: Type of embedding used, 'sin_cos' or 'learned'
            name: Name for this op
        """
        assert embedding_type in ['sin_cos', 'learned']  # error checking
        super(PatchEncoder, self).__init__(name=name, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.embedding_type = embedding_type

        self.projection = Dense(units=projection_dim)
        if embedding_type == 'learned':
            self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)
        else:
            self.position_embedding = positional_encoding(num_patches, projection_dim)

    def call(self, patch, **kwargs):
        if self.embedding_type == 'learned':
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            embedding = self.position_embedding(positions)
        else:
            embedding = self.position_embedding
        encoded = self.projection(patch) + embedding
        return encoded

    def get_config(self):
        cfg = super(PatchEncoder, self).get_config()
        cfg.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
            'embedding_type': self.embedding_type
        })
        return cfg


class PatchDecoder(Layer):
    def __init__(self, patch_width, patch_height, x_patches, y_patches, channels=1, ignore_last=False, name=None, **kwargs):
        super(PatchDecoder, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.x_patches = x_patches
        self.y_patches = y_patches
        self.ignore_last = ignore_last

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
        if self.ignore_last: encoded = encoded[:, :-1, :]
        reshaped = tf.reshape(encoded, (-1, self.y_patches, self.x_patches, self.patch_height, self.patch_width))
        reshaped = tf.transpose(reshaped, [0, 1, 3, 2, 4])
        reshaped = tf.reshape(reshaped, (-1, self.y_patches * self.patch_height, self.x_patches * self.patch_width, self.channels))

        # Merging into final output
        # def func(x):
        #     x = tf.nn.space_to_depth(x, self.patch_width)
        #     x = tf.reshape(x, [self.x_patches * self.patch_width, self.y_patches * self.patch_height, 1])
        #     return x

        return reshaped

    def get_config(self):
        cfg = super(PatchDecoder, self).get_config()
        cfg.update({
            'patch_width': self.patch_width,
            'patch_height': self.patch_height,
            'x_patches': self.x_patches,
            'y_patches': self.y_patches
        })
        return cfg


def Transformer(input_shape=(256, 256, 1), sinogram_height=1, sinogram_width=256, dim=256, layers=16, num_heads=16,
                mlp_units=512, output_patch_height=16, output_patch_width=16, output_x_patches=16, output_y_patches=16,
                dropout=0., out_activation='linear', activation='gelu', norm=partial(LayerNormalization, epsilon=1e-5)):
    # error checking
    if input_shape[0] % sinogram_height != 0 or input_shape[1] % sinogram_width != 0:
        raise ValueError("Cannot divide image into even patches")

    num_patches = int(input_shape[1] / sinogram_width * input_shape[0] / sinogram_height)

    inputs = Input(shape=input_shape)

    # patch
    x = Patches(patch_height=sinogram_height, patch_width=sinogram_width, name="patchify")(inputs)
    x = PatchEncoder(num_patches=num_patches, projection_dim=dim, name="projection")(x)

    for i in range(layers):
        x = EncoderBlock(num_heads=num_heads, mlp_units=mlp_units, dim=dim, dropout=dropout, activation=activation,
                         norm=norm, name=f"block_{i}")(x)

    x = Dense(output_patch_height * output_patch_width, activation=out_activation, name='out_projection')(x)

    # reshape
    x = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches, name="depatchify")(x)

    return Model(inputs=inputs, outputs=x)


def CTransformer(input_shape=(256, 256, 1), sinogram_height=1, sinogram_width=256, enc_dim=256, enc_layers=8,
                 enc_mlp_units=512, enc_heads=16, dec_projection=True, dec_dim=256, dec_layers=8, dec_heads=16,
                 dec_mlp_units=512, output_projection=False, output_patch_height=16, output_patch_width=16,
                 output_x_patches=16, output_y_patches=16, dropout=0.,
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
        x = Dense(output_patch_height * output_patch_width, name='out_projection')(x)

    # reshape
    x = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches, name="depatchify")(x)

    return Model(inputs=inputs, outputs=x)


def CTranslator(input_shape=(256, 256, 1), sinogram_height=1, sinogram_width=256, enc_dim=512, enc_layers=8,
                enc_mlp_units=2048, enc_heads=16, dec_dim=512, dec_layers=8, dec_heads=16,
                dec_mlp_units=2048, output_projection=True, output_patch_height=16, output_patch_width=16,
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


class CTransformerModel(Model):
    def __init__(self, input_shape=(256, 256, 1), sinogram_height=1, sinogram_width=256, enc_dim=256, enc_layers=8,
                 enc_mlp_units=512, enc_heads=16, dec_projection=True, dec_dim=256, dec_layers=8, dec_heads=16,
                 dec_mlp_units=512, output_projection=True, output_patch_height=16, output_patch_width=16,
                 output_x_patches=16, output_y_patches=16, dropout=0., activation='gelu',
                 norm=partial(LayerNormalization, epsilon=1e-5), name='mae',
                 radon=False, radon_transform=None, dose=4096, final_shape=(362, 362, 1)):
        super(CTransformerModel, self).__init__(name=name)

        self.radon = radon
        self.radon_transform = radon_transform

        self.dose = dose

        self.inp_shape = input_shape
        self.sinogram_height = sinogram_height
        self.sinogram_width = sinogram_width
        self.enc_dim = enc_dim
        self.enc_layers = enc_layers
        self.enc_mlp_units = enc_mlp_units
        self.enc_heads = enc_heads
        self.dec_projection = dec_projection
        self.dec_dim = dec_dim
        self.dec_layers = dec_layers
        self.dec_heads = dec_heads
        self.dec_mlp_units = dec_mlp_units
        self.output_projection = output_projection
        self.dropout = dropout
        self.activation = activation

        self.output_patch_height = output_patch_height
        self.output_patch_width = output_patch_width
        self.output_x_patches = output_x_patches
        self.output_y_patches = output_y_patches

        self.final_shape = final_shape

        if input_shape[0] % sinogram_height != 0 or input_shape[1] % sinogram_width != 0:
            raise ValueError("Cannot divide image into even patches")

        self.num_patches = int(input_shape[1] / sinogram_width * input_shape[0] / sinogram_height)

        self.patches = Patches(sinogram_width, sinogram_height, f'{name}_patches')
        self.patches_2 = Patches(output_patch_width, output_patch_height, name=f'{name}_patches_2')

        self.patch_encoder = PatchEncoder(num_patches=self.num_patches,
                                          projection_dim=enc_dim, name="enc_projection")

        self.enc_blocks = [
            EncoderBlock(enc_heads, enc_dim, enc_mlp_units, dropout, activation=activation,
                         norm=norm, name=f'{name}_enc_block_{i}')
            for i in range(enc_layers)
        ]

        self.enc_norm = norm(name=f'{name}_enc_norm')

        self.dec_projection_layer = PatchEncoder(num_patches=self.num_patches,
                                                 projection_dim=self.dec_dim, name="dec_projection")
        # Dense(dec_dim, name=f'{name}_dec_projection')

        self.dec_blocks = [
            EncoderBlock(num_heads=dec_heads, mlp_units=dec_mlp_units, dim=dec_dim, dropout=dropout,
                         activation=activation, norm=norm, name=f"{name}_dec_block_{i}")
            for i in range(dec_layers)
        ]

        self.dec_norm = norm(name=f'{name}_dec_norm')

        self.output_projection_layer = Dense(output_patch_height * output_patch_width, name=f'{name}_output_projection')

        self.depatchify = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches,
                                       name=f"{name}_depatchify")

    def call(self, inputs, training=None, mask=None):
        # patch
        x = self.patches(inputs)
        x = self.patch_encoder(x)

        # encoder
        for enc_layer in self.enc_blocks:
            x = enc_layer(x)

        # decoder projection
        if self.dec_projection:
            x = self.dec_projection_layer(x)

        # decoder
        for dec_layer in self.dec_blocks:
            x = dec_layer(x)

        # output projection
        if self.output_projection:
            x = self.output_projection_layer(x)

        # reshape
        x = self.depatchify(x)
        return x

    def train_step(self, data):
        x, y = data

        if self.radon:
            # apply radon transform
            sinogram = self.radon_transform(tf.image.resize(tf.expand_dims(y, axis=-1), (362, 362)), training=False)
            sinogram = tf.clip_by_value(sinogram, 0, 10) * 0.46451485

            # preprocess data
            sinogram = add_noise(sinogram, dose=self.dose)
            sinogram, y = preprocess_data(sinogram[:, ::-1, ::-1, -1], y, resize_img=False)
        else:
            # preprocess data
            sinogram, y = preprocess_data(x, y, resize_img=False)

        with tf.GradientTape() as tape:
            y_pred = self(sinogram, training=True)
            # y_pred = tf.image.resize(y_pred, self.final_shape[:-1])

            # y = tf.image.resize(y, self.final_shape[:-1])

            loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return self.compute_metrics(x, y, y_pred, None)

    def test_step(self, data):
        x, y = data

        # preprocess data
        sinogram, y = preprocess_data(x, y, resize_img=False)
        print(sinogram.shape)
        # y = tf.image.resize(y, self.final_shape[:-1])

        # call model
        y_pred = self(sinogram, training=False)
        # y_pred = tf.image.resize(y_pred, self.final_shape[:-1])

        # evaluate loss
        self.compute_loss(x, y, y_pred, None)

        return self.compute_metrics(x, y, y_pred, None)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        norm_cls = deserialize(config['norm']).__class__
        del config['norm']['config']['name']
        norm = partial(norm_cls, **config['norm']['config'])
        del config['norm']
        radon_transform = LPRadonForward(**config['radon_transform'])
        del config['radon_transform']

        return cls(**config, norm=norm, radon_transform=radon_transform)

    def get_config(self):
        cfg = super(CTransformerModel, self).get_config()
        cfg.update({
            'input_shape': self.inp_shape,
            'sinogram_height': self.sinogram_height,
            'sinogram_width': self.sinogram_width,
            'enc_dim': self.enc_dim,
            'enc_layers': self.enc_layers,
            'enc_mlp_units': self.enc_mlp_units,
            'enc_heads': self.enc_heads,
            'dec_layers': self.dec_layers,
            'dec_heads': self.dec_heads,
            'dec_mlp_units': self.dec_mlp_units,
            'dropout': self.dropout,
            'activation': self.activation,
            'mask_ratio': self.mask_ratio,
            'norm': serialize(self.enc_blocks[0].norm1),
            'output_patch_height': self.output_patch_height,
            'output_patch_width': self.output_patch_width,
            'output_x_patches': self.output_x_patches,
            'output_y_patches': self.output_y_patches,
            'radon': self.radon,
            'radon_transform': self.radon_transform.get_config(),
            'dose': self.dose,
            'final_shape': self.final_shape
        })
        return cfg