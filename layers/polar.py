import tensorflow as tf

keras = tf.keras

from keras.models import *
from keras.layers import *
from functools import partial

from layers.vision_transformer import PatchEncoder, DecoderBlock
from layers.utils import Cart2Polar, Polar2Cart

from model import create_model

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
