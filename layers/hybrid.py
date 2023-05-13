import tensorflow as tf

from data_generation.lpradon import LPRadonForward, LPRadonFBP
from utils import add_noise, preprocess_data

keras = tf.keras

from keras.models import *
from keras.layers import *
from functools import partial

from layers.vision_transformer import Patches, PatchEncoder, DecoderBlock, PatchDecoder


def DenoiseCT(mae, num_mask=0, dec_dim=256, dec_layers=8, dec_heads=16, dec_mlp_units=512, output_projection=False,
              output_patch_height=16, output_patch_width=16, output_x_patches=16, output_y_patches=16,
              norm=partial(LayerNormalization, epsilon=1e-5)):
    input_shape = mae.inp_shape
    num_patches = mae.num_patches

    inputs = [Input(input_shape), Input(num_mask, dtype=tf.int32), Input(num_patches - num_mask, dtype=tf.int32),
              Input((output_patch_height * output_y_patches, output_patch_width * output_x_patches, 1))]

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

    y = Patches(output_patch_width, output_patch_height, name='dec_patches')(y)
    y = PatchEncoder(output_y_patches * output_x_patches, dec_dim, embedding_type='learned', name='dec_projection')(y)

    for i in range(dec_layers):
        y = DecoderBlock(dec_heads, mae.enc_dim, dec_dim, mlp_units=dec_mlp_units, num_patches=num_patches,
                         dropout=mae.dropout, activation=mae.activation, norm=norm, name=f'dec_block_{i}')((y, x))

    y = norm(name='output_norm')(y)
    y = Dense(output_patch_height * output_patch_width, name='output_projection')(y)
    y = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches)(y)

    return Model(inputs=inputs, outputs=y)


class DenoiseCTModel(Model):
    def __init__(self, mae, num_mask=0, dec_dim=256, dec_layers=8, dec_heads=16, dec_mlp_units=512,
                 output_patch_height=16, output_patch_width=16, output_x_patches=32, output_y_patches=32,
                 norm=partial(LayerNormalization, epsilon=1e-5), radon=True,
                 radon_transform=None, fbp=None, dosage=4096, final_shape=(362, 362, 1), *args, **kwargs):
        super(DenoiseCTModel, self).__init__(*args, **kwargs)
        self.num_mask = num_mask

        self.dec_dim = dec_dim
        self.dec_layers = dec_layers
        self.dec_heads = dec_heads
        self.dec_mlp_units = dec_mlp_units

        self.output_patch_height = output_patch_height
        self.output_patch_width = output_patch_width
        self.output_x_patches = output_x_patches
        self.output_y_patches = output_y_patches

        self.norm = norm

        self.radon = radon
        self.radon_transform = radon_transform
        self.fbp = fbp
        self.dose = dosage

        self.final_shape = final_shape
        self.inp_shape = mae.inp_shape
        self.mae = mae

        self.num_patches = mae.num_patches

        self.patches = Patches(self.output_patch_width, self.output_patch_height, name='dec_patches')
        self.patch_encoder = PatchEncoder(self.output_y_patches * self.output_x_patches, self.dec_dim,
                                          embedding_type='learned', name='dec_projection')

        self.dec_blocks = [
            DecoderBlock(self.dec_heads, self.mae.enc_dim, self.dec_dim, mlp_units=self.dec_mlp_units,
                         num_patches=self.num_patches, dropout=self.mae.dropout, activation=self.mae.activation,
                         norm=self.norm, name=f'dec_block_{i}')
            for i in range(self.dec_layers)
        ]

        self.norm_layer = self.norm(name='output_norm')

        self.dense = Dense(self.output_patch_height * self.output_patch_width, name='output_projection')
        self.patch_decoder = PatchDecoder(self.output_patch_width, self.output_patch_height,
                                          self.output_x_patches, self.output_y_patches)

        self.resize = Resizing(
            final_shape[0], final_shape[1]
        )

    def call(self, inputs, training=None, mask=None):
        x, mask_indices, unmask_indices, y = inputs

        self.mae.patches.trainable = False
        x = self.mae.patches(x)

        self.mae.patch_encoder.trainable = False
        self.mae.patch_encoder.num_mask = self.num_mask

        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.mae.patch_encoder(x, mask_indices, unmask_indices)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = unmasked_embeddings

        for enc_block in self.mae.enc_blocks:
            enc_block.trainable = False
            encoder_outputs = enc_block(encoder_outputs)

        self.mae.enc_norm.trainable = False
        encoder_outputs = self.mae.enc_norm(encoder_outputs)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        x = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        y = self.patches(y)
        y = self.patch_encoder(y)

        for dec_block in self.dec_blocks:
            y = dec_block((y, x))

        y = self.norm_layer(y)
        y = self.dense(y)
        y = self.patch_decoder(y)

        return self.resize(y)

    def train_step(self, data):
        x, y = data

        if self.radon:
            # apply radon transform
            sinogram = self.radon_transform(tf.image.resize(y, (362, 362)), training=False)
            sinogram = tf.clip_by_value(sinogram, 0, 10) * 0.46451485

            # preprocess data
            sinogram = add_noise(sinogram, dose=self.dose)
            fbp = self.fbp(sinogram, training=False) * 750  # todo may need to do some postprocessing on fbp
            fbp = tf.image.central_crop(fbp, self.final_shape[0] / self.inp_shape[0])

            sinogram, y = preprocess_data(sinogram[:, ::-1, ::-1, :], y, resize_img=False, expand_dims=False)
            _, fbp = preprocess_data(sinogram, fbp, resize_img=True, expand_dims=False)
            y = self.resize(y)

            x = (sinogram, x[1], x[2], fbp)
        else:
            # preprocess data
            sinogram, y = preprocess_data(x[0], y, resize_img=False, expand_dims=False)
            x = (sinogram, x[1], x[2], x[3])

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        # preprocess data
        sinogram, y = preprocess_data(x[0], y, resize_img=False, expand_dims=False)
        _, fbp = preprocess_data(sinogram, x[3], resize_img=True, expand_dims=False)
        # fbp = tf.image.central_crop(fbp, self.final_shape[0] / self.inp_shape[0])

        # call model
        y_pred = self((sinogram, x[1], x[2], fbp), training=False)

        # evaluate loss
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        norm_cls = deserialize(config['norm']).__class__
        del config['norm']['config']['name']
        norm = partial(norm_cls, **config['norm']['config'])
        del config['norm']
        radon_transform = LPRadonForward(**config['radon_transform'])
        del config['radon_transform']
        fbp = LPRadonFBP(**config['fbp'])
        del config['fbp']

        return cls(**config, norm=norm, radon_transform=radon_transform, fbp=fbp)

    def get_config(self):
        cfg = super(DenoiseCTModel, self).get_config()
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
            'fbp': self.fbp.get_config(),
            'dose': self.dose,
        })
        return cfg
