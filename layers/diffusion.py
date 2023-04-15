import math

import tensorflow as tf

keras = tf.keras

from keras.models import *
from keras.layers import *
from functools import partial

from layers.vision_transformer import Patches, PatchEncoder, DecoderBlock, PatchDecoder, MLP
from layers.utils import Cart2Polar, Polar2Cart

from model import create_model


class DiTBlock(Layer):
    def __init__(self, num_heads=16, dim=256, mlp_units=512, dropout=0., out_dim=None, activation='gelu',
                 name='vit_block', norm=partial(LayerNormalization, epsilon=1e-5), zero_init=False, **kwargs):
        super().__init__(name=name, **kwargs)
        if out_dim is None:
            out_dim = dim

        self.num_heads = num_heads
        self.dim = dim
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.out_dim = out_dim
        self.activation = activation
        self.zero_init = zero_init

        self.norm1 = norm(name=f"{name}_norm_1")
        self.mhsa = MultiHeadAttention(num_heads=num_heads, key_dim=dim, dropout=dropout, name=f"{name}_mha")
        self.norm2 = norm(name=f"{name}_norm_2")
        self.mlp = MLP(hidden_dim=mlp_units, out_dim=out_dim, hidden_activation=activation, name=f"{name}_mlp")

        # idk why the swish comes first, ask https://github.com/facebookresearch/DiT/blob/main/models.py
        self.flatten = Flatten()
        if self.zero_init:
            self.adaln = Sequential([
                Activation("swish"),
                Dense(6 * out_dim, kernel_initializer=tf.keras.initializers.Zeros())
            ])
        else:
            self.adaln = Sequential([
                Activation("swish"),
                Dense(6 * out_dim)
            ])

    def modulate(self, x, shift, scale):
        return x * (1 + tf.expand_dims(scale, axis=1)) + tf.expand_dims(shift, axis=1)

    def call(self, inputs, **kwargs):
        x, c = inputs  # split into conditioning the actual thing

        # adaln stuff
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = tf.split(self.adaln(c), 6, axis=1)

        # transformer stuff
        if self.zero_init:
            x1 = self.norm1(x)
            x1 = self.modulate(x1, shift_msa, scale_msa)
            attn = self.mhsa(x1, x1)
            x2 = attn * tf.expand_dims(gate_msa, axis=1) + x1
            x3 = self.norm2(x2)
            x3 = self.modulate(x3, shift_mlp, scale_mlp)
            x3 = self.mlp(x3)
            return x2 + x3 * tf.expand_dims(gate_mlp, axis=1)
        else:
            x1 = self.norm1(x)
            x1 = self.modulate(x1, shift_msa, scale_msa)
            attn = self.mhsa(x1, x1)
            x2 = attn + x1
            x3 = self.norm2(x2)
            x3 = self.modulate(x3, shift_mlp, scale_mlp)
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
        cfg = super(DiTBlock, self).get_config()
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


class TimeEmbedding(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def DiffusionTransformer(
    mae, num_mask=0, dec_dim=256, dec_layers=8, dec_heads=16, dec_mlp_units=512, output_patch_height=16,
    output_patch_width=16, output_x_patches=16, output_y_patches=16,
    norm=partial(LayerNormalization, epsilon=1e-5), timestep_embedding="mlp", conditioning="cross-attention"
):
    input_shape = mae.inp_shape

    num_patches = mae.num_patches

    inputs = [Input(input_shape), Input(num_mask, dtype=tf.int32), Input(num_patches, dtype=tf.int32),
              Input((output_patch_height * output_y_patches, output_patch_width * output_x_patches, 1),
                    dtype='float32'),
              Input((1,), dtype=tf.float32)]

    x, mask_indices, unmask_indices, y, tt = inputs

    if timestep_embedding == "sin-cos":
        time_token = TimeEmbedding(dec_dim)(tt[:, 0])
    else: time_token = Dense(dec_dim)(Dense(100, activation="gelu")(tt))

    mae.patches.trainable = False
    x = mae.patches(x)

    mae.patch_encoder.trainable = False
    mae.patch_encoder.num_mask = num_mask
    (
        unmasked_embeddings,
        masked_embeddings,
        unmasked_positions,
        mask_indices,
        unmask_indices,
    ) = mae.patch_encoder(x, mask_indices, unmask_indices)

    # Pass the unmasked patches to the encoder.
    encoder_outputs = unmasked_embeddings

    for enc_block in mae.enc_blocks:
        enc_block.trainable = False
        encoder_outputs = enc_block(encoder_outputs)

    mae.enc_norm.trainable = False
    encoder_outputs = mae.enc_norm(encoder_outputs)

    # Create the decoder inputs.
    encoder_outputs = encoder_outputs + unmasked_positions
    x = tf.concat([encoder_outputs, masked_embeddings], axis=1)
    x = concatenate(
        [
            GlobalAveragePooling1D()(x),
            TimeEmbedding(input_shape[1])(tt[:, 0])
        ], axis=1
    )

    y = Patches(output_patch_width, output_patch_height, name='dec_patches')(y)
    y = PatchEncoder(output_y_patches * output_x_patches, dec_dim, embedding_type='learned', name='dec_projection')(y)

    time_token = tf.expand_dims(time_token, axis=1)
    y = concatenate([y, time_token], axis=1)

    if conditioning == "cross-attention":
        for i in range(dec_layers):
            y = DecoderBlock(dec_heads, mae.enc_dim, dec_dim, mlp_units=dec_mlp_units, num_patches=num_patches + 1,
                             dropout=mae.dropout, activation=mae.activation, norm=norm, name=f'dec_block_{i}')((y, x))
    elif conditioning == "adaln" or conditioning == "adaln-zero":
        for i in range(dec_layers):
            y = DiTBlock(
                dec_heads, dec_dim, mlp_units=dec_mlp_units,
                dropout=mae.dropout, activation=mae.activation, norm=norm, name=f'dit_block_{i}',
                zero_init=conditioning == "adaln-zero"
            )((y, x))

    y = norm(name='output_norm')(y)
    y = Dense(output_patch_height * output_patch_width, name='output_projection')(y)
    y = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches,
                     ignore_last=True)(y)

    return Model(inputs=inputs, outputs=y)


def CircleTransformer(mae, num_mask=0, dec_dim=256, dec_layers=8, dec_heads=16, dec_mlp_units=512,
                      output_width=362, output_height=362, norm=partial(LayerNormalization, epsilon=1e-5)):
    input_shape = mae.inp_shape
    num_patches = mae.num_patches

    inputs = [Input(input_shape), Input(num_mask, dtype=tf.int32), Input(num_patches - num_mask, dtype=tf.int32),
              Input((output_height, output_width, 1), dtype='float32')]

    x, mask_indices, unmask_indices, y = inputs

    mae.patches.trainable = False
    x = mae.patches(x)

    mae.patch_encoder.trainable = False
    mae.patch_encoder.num_mask = num_mask
    (
        unmasked_embeddings,
        masked_embeddings,
        unmasked_positions,
        mask_indices,
        unmask_indices,
    ) = mae.patch_encoder(x, mask_indices, unmask_indices)

    # Pass the unmaksed patche to the encoder.
    encoder_outputs = unmasked_embeddings

    for enc_block in mae.enc_blocks:
        enc_block.trainable = False
        encoder_outputs = enc_block(encoder_outputs)

    mae.enc_norm.trainable = False
    encoder_outputs = mae.enc_norm(encoder_outputs)

    # Create the decoder inputs.
    encoder_outputs = encoder_outputs + unmasked_positions
    x = tf.concat([encoder_outputs, masked_embeddings], axis=1)

    y = Cart2Polar(num_patches, dec_dim)(y)
    y = tf.reshape(y, (-1, num_patches, dec_dim))
    y = PatchEncoder(num_patches, dec_dim, embedding_type='learned', name='dec_projection')(y)

    for i in range(dec_layers):
        y = DecoderBlock(dec_heads, mae.enc_dim, dec_dim, mlp_units=dec_mlp_units, num_patches=num_patches,
                         dropout=mae.dropout, activation=mae.activation, norm=norm, name=f'dec_block_{i}')((y, x))

    y = norm(name='output_norm')(y)
    y = Dense(dec_dim, name='output_projection')(y)
    y = tf.expand_dims(y, -1)
    y = Polar2Cart(output_height, output_width)(y)

    return Model(inputs=inputs, outputs=y)


def CircleUNet(input_shape=(362, 1000, 1), output_width=362, output_height=362):
    inputs = Input(input_shape)

    # Create the decoder inputs.
    x = Cart2Polar(1024, 512)(inputs)

    model = create_model(
        {
            "task": "sparse",
            "shape": (1024, 512, 1),
            "blocks": (1, 2, 2, 3, 4),
            "filters": (64, 64, 64, 64, 64),
            "activation": "swish",
            "kernel_size": 7,
            "drop_connect_rate": 0.05,
            "dropout_rate": 0.05,
            "block_size": 10,
            "noise": 0.20,
            "dropblock_3d": False,
            "dropblock_2d": True,
            "block_type": "convnext",
            "attention": "gc",
            "dimensions": 2,
            "initial_dimensions": 2,
            "final_activation": "linear",
            "final_filters": 1
        }
    )
    x = model(x)

    x = Polar2Cart(output_height, output_width)(x)
    return Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    from layers.mae import MAE

    mae = MAE(
        enc_layers=4,
        dec_layers=1,
        sinogram_width=513,
        sinogram_height=1,
        input_shape=(1024, 513, 1),
        enc_dim=512,
        enc_mlp_units=2048,
        dec_dim=512,
        dec_mlp_units=2048
    )

    model = DiffusionTransformer(
        mae,
        num_mask=0,
        dec_dim=256,
        dec_mlp_units=1024,
        dec_layers=4,
        output_patch_width=16,
        output_patch_height=16,
        output_x_patches=32,
        output_y_patches=32,
        conditioning="adaln-zero"
    )
    model.summary()

    model(
        [
            tf.zeros((1, 1024, 513, 1)),
            tf.zeros((1, 0)),
            tf.zeros((1, 1024)),
            tf.zeros((1, 512, 512, 1)),
            tf.zeros((1, 1))
        ]
    )
