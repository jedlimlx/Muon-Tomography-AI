import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from functools import partial
from layers.vision_transformer import Patches, PatchEncoder, DecoderBlock, PatchDecoder


def DenoiseCT(mae, num_mask=0, dec_dim=256, dec_layers=8, dec_heads=16, dec_mlp_units=512, output_projection=False,
              output_patch_height=16, output_patch_width=16, output_x_patches=16, output_y_patches=16,
              norm=partial(LayerNormalization, epsilon=1e-5)):
    input_shape = mae.inp_shape
    num_patches = mae.num_patches

    inputs = [Input(input_shape), Input(num_mask, dtype=tf.int32), Input(num_patches - num_mask, dtype=tf.int32),
              Input((output_patch_height * output_y_patches,
                     output_patch_width * output_x_patches, 1), dtype='float32')]

    x, mask_indices, unmask_indices, y = inputs

    print(y.shape)

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

    y = Patches(output_patch_width, output_patch_height, name='dec_patches')(y)
    y = PatchEncoder(output_y_patches * output_x_patches, dec_dim, embedding_type='learned', name='dec_projection')(y)

    for i in range(dec_layers):
        y = DecoderBlock(dec_heads, mae.enc_dim, dec_dim, mlp_units=dec_mlp_units, num_patches=num_patches,
                         dropout=mae.dropout, activation=mae.activation, norm=norm, name=f'dec_block_{i}')((y, x))

    y = norm(name='output_norm')(y)
    y = Dense(output_patch_height * output_patch_width, name='output_projection')(y)
    y = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches)(y)

    return Model(inputs=inputs, outputs=y)

