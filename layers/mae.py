from functools import partial

import tensorflow as tf

keras = tf.keras

from keras.layers import *
from keras.models import *

from layers.vision_transformer import Patches, positional_encoding, EncoderBlock, PatchDecoder, PatchEncoder

from utils import add_noise, preprocess_data


class MAEPatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, embedding_type='sin_cos', mask_proportion=0.75, name=None,
                 downstream=False, **kwargs):
        """
        Projection and embedding
        Args:
            num_patches: Sequence length
            projection_dim: Output dim after projection
            embedding_type: Type of embedding used, 'sin_cos' or 'learned'
            name: Name for this op
        """
        assert embedding_type in ['sin_cos', 'learned']  # error checking
        super(MAEPatchEncoder, self).__init__(name=name, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.embedding_type = embedding_type
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        self.num_mask = int(self.num_patches * self.mask_proportion)

        self.projection = Dense(units=projection_dim)
        if embedding_type == 'learned':
            self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)
        else:
            self.position_embedding = positional_encoding(num_patches, projection_dim)

        self.mask_token = None

    def build(self, input_shape):
        _, depth, area = input_shape
        self.mask_token = self.add_weight(shape=(1, area), initializer='random_uniform', name=f'{self.name}_mask_token')

    def call(self, patch, mask_indices=None, unmask_indices=None, **kwargs):
        batch_size = tf.shape(patch)[0]
        if self.embedding_type == 'learned':
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            embedding = self.position_embedding(positions)
        else:
            embedding = self.position_embedding

        embedding = tf.tile(embedding, (batch_size, 1, 1))
        encoded = self.projection(patch) + embedding

        if mask_indices is None or unmask_indices is None:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)

        # The encoder input is the unmasked patch embeddings. Here we gather
        # all the patches that should be unmasked.
        unmasked_embeddings = tf.gather(
            encoded, unmask_indices, axis=1, batch_dims=1
        )  # (B, unmask_numbers, projection_dim)

        # Get the unmasked and masked position embeddings. We will need them
        # for the decoder.
        unmasked_positions = tf.gather(
            embedding, unmask_indices, axis=1, batch_dims=1
        )  # (B, unmask_numbers, projection_dim)
        masked_positions = tf.gather(
            embedding, mask_indices, axis=1, batch_dims=1
        )  # (B, mask_numbers, projection_dim)

        # Repeat the mask token number of mask times.
        # Mask tokens replace the masks of the image.
        mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
        mask_tokens = tf.repeat(
            mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
        )

        # Get the masked embeddings for the tokens.
        masked_embeddings = self.projection(mask_tokens) + masked_positions
        return (
            unmasked_embeddings,  # Input to the encoder.
            masked_embeddings,  # First part of input to the decoder.
            unmasked_positions,  # Added to the encoder outputs.
            mask_indices,  # The indices that were masked.
            unmask_indices,  # The indices that were unmasked.
        )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask:]
        return mask_indices, unmask_indices

    def get_config(self):
        cfg = super(MAEPatchEncoder, self).get_config()
        cfg.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
            'embedding_type': self.embedding_type,
            'mask_proportion': self.mask_proportion,
            'downstream': self.downstream
        })
        return cfg


class MAE(Model):
    def __init__(self, input_shape=(256, 256, 1), sinogram_height=1, sinogram_width=256, enc_dim=256, enc_layers=8,
                 enc_mlp_units=512, enc_heads=16, dec_dim=256, dec_layers=8, dec_heads=16,
                 dec_mlp_units=512, dropout=0., activation='gelu', mask_ratio=0.75,
                 norm=partial(LayerNormalization, epsilon=1e-5), name='mae',
                 radon=False, radon_transform=None, dose=4096, denoise=False, apply_noise=False):
        super(MAE, self).__init__(name=name)

        self.radon = radon
        self.radon_transform = radon_transform

        self.dose = dose
        self.denoise = denoise
        self.apply_noise = apply_noise

        self.inp_shape = input_shape
        self.sinogram_height = sinogram_height
        self.sinogram_width = sinogram_width
        self.enc_dim = enc_dim
        self.enc_layers = enc_layers
        self.enc_mlp_units = enc_mlp_units
        self.enc_heads = enc_heads
        self.dec_dim = dec_dim
        self.dec_layers = dec_layers
        self.dec_heads = dec_heads
        self.dec_mlp_units = dec_mlp_units
        self.dropout = dropout
        self.activation = activation
        self.mask_ratio = mask_ratio

        if input_shape[0] % sinogram_height != 0 or input_shape[1] % sinogram_width != 0:
            raise ValueError("Cannot divide image into even patches")

        self.num_patches = int(input_shape[1] / sinogram_width * input_shape[0] / sinogram_height)

        self.patches = Patches(sinogram_width, sinogram_height, f'{name}_patches')
        self.patch_encoder = MAEPatchEncoder(self.num_patches, enc_dim, mask_proportion=mask_ratio,
                                             name=f'{name}_enc_projection')

        self.enc_blocks = [
            EncoderBlock(enc_heads, enc_dim, enc_mlp_units, dropout, activation=activation,
                         norm=norm, name=f'{name}_enc_block_{i}')
            for i in range(enc_layers)
        ]

        self.enc_norm = norm(name=f'{name}_enc_norm')

        self.dec_projection = Dense(dec_dim, name=f'{name}_dec_projection')

        self.dec_blocks = [
            EncoderBlock(dec_heads, dec_dim, dec_mlp_units, dropout, activation=activation,
                         norm=norm, name=f'{name}_dec_block_{i}')
            for i in range(dec_layers)
        ]

        self.dec_norm = norm(name=f'{name}_dec_norm')

        self.output_projection = Dense(sinogram_width * sinogram_height, name=f'{name}_output_projection')

        self.depatchify = PatchDecoder(sinogram_width, sinogram_height, int(input_shape[1] / sinogram_width),
                                       int(input_shape[0] / sinogram_height), name=f'{name}_depatchify')

    def call(self, inputs, denoised_inputs=None, training=None, mask=None):
        patches = self.patches(inputs)

        if denoised_inputs is None:
            denoised_patches = self.patches(inputs)
        else:
            denoised_patches = self.patches(denoised_inputs)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmasked patches to the encoder.
        encoder_outputs = unmasked_embeddings

        for enc_block in self.enc_blocks:
            encoder_outputs = enc_block(encoder_outputs)

        encoder_outputs = self.enc_norm(encoder_outputs)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.dec_projection(decoder_inputs)
        for dec_block in self.dec_blocks:
            decoder_outputs = dec_block(decoder_outputs)

        decoder_outputs = self.dec_norm(decoder_outputs)

        decoder_outputs = self.output_projection(decoder_outputs)

        """
        decoder_outputs = tf.scatter_nd(
            indices=tf.expand_dims(unmask_indices, axis=-1),
            updates=tf.reshape(decoder_outputs, (-1, tf.shape(decoder_outputs)[2])),
            shape=(tf.shape(decoder_outputs)[0], self.num_patches, tf.shape(decoder_outputs)[2])
        ) + tf.scatter_nd(
            indices=tf.expand_dims(mask_indices, axis=-1),
            updates=tf.reshape(decoder_outputs, (-1, tf.shape(decoder_outputs)[2])),
            shape=(tf.shape(decoder_outputs)[0], self.num_patches, tf.shape(decoder_outputs)[2])
        )
        """

        decoder_patches = self.depatchify(decoder_outputs)

        return denoised_patches, decoder_patches, mask_indices, unmask_indices

    def train_step(self, data):
        def process_sinogram(sinogram):
            sinogram = tf.clip_by_value(sinogram, 0, 10)
            sinogram = sinogram[:, ::-1, ::-1, :] * 0.46451485

            sinogram = (sinogram - 0.030857524) / 0.023017514
            sinogram = tf.image.resize(sinogram, (1024, 513))
            return sinogram

        if self.radon:
            sinogram = self.radon_transform(data, training=False)

            if self.apply_noise:
                noised_sinogram = add_noise(sinogram, dose=self.dose)
            else:
                noised_sinogram = sinogram

            sinogram = process_sinogram(sinogram)
            noised_sinogram = process_sinogram(noised_sinogram)

            data = sinogram
            noised_data = noised_sinogram

            if self.denoise:
                data = noised_sinogram
        else:
            noised_data = data

        with tf.GradientTape() as tape:
            patches, decoder_patches, mask_indices, unmasked_indices = self(noised_data, denoised_inputs=data)
            loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
            loss_output = decoder_patches[:, int(self.num_patches * (1 - self.mask_ratio)):]
            total_loss = self.compiled_loss(loss_patch, loss_output)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(loss_patch, loss_output)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        patches, decoder_patches, mask_indices, unmasked_indices = self(data)
        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = decoder_patches[:, int(self.num_patches * (1 - self.mask_ratio)):]

        self.compiled_loss(loss_patch, loss_output)

        self.compiled_metrics.update_state(loss_patch, loss_output)

        return {m.name: m.result() for m in self.metrics}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        norm_cls = deserialize(config['norm']).__class__
        del config['norm']['config']['name']
        norm = partial(norm_cls, **config['norm']['config'])
        del config['norm']

        return cls(**config, norm=norm)

    def get_config(self):
        cfg = super(MAE, self).get_config()
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
            'norm': serialize(self.enc_blocks[0].norm1)
        })
        return cfg


class MaskedCTransformer(Model):
    def __init__(self, input_shape=(256, 256, 1), sinogram_height=1, sinogram_width=256, enc_dim=256, enc_layers=8,
                 enc_mlp_units=512, enc_heads=16, dec_dim=256, dec_layers=8, dec_heads=16,
                 dec_mlp_units=512, output_patch_height=16, output_patch_width=16,
                 output_x_patches=16, output_y_patches=16, dropout=0., activation='gelu', mask_ratio=0.75,
                 norm=partial(LayerNormalization, epsilon=1e-5), name='mae',
                 radon=False, radon_transform=None, dose=4096):
        super(MaskedCTransformer, self).__init__(name=name)

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
        self.dec_dim = dec_dim
        self.dec_layers = dec_layers
        self.dec_heads = dec_heads
        self.dec_mlp_units = dec_mlp_units
        self.dropout = dropout
        self.activation = activation
        self.mask_ratio = mask_ratio

        self.output_patch_height = output_patch_height
        self.output_patch_width = output_patch_width
        self.output_x_patches = output_x_patches
        self.output_y_patches = output_y_patches

        if input_shape[0] % sinogram_height != 0 or input_shape[1] % sinogram_width != 0:
            raise ValueError("Cannot divide image into even patches")

        self.num_patches = int(input_shape[1] / sinogram_width * input_shape[0] / sinogram_height)

        self.patches = Patches(sinogram_width, sinogram_height, f'{name}_patches')
        self.patches_2 = Patches(output_patch_width, output_patch_height, name=f'{name}_patches_2')
        self.patch_encoder = MAEPatchEncoder(self.num_patches, enc_dim, mask_proportion=mask_ratio,
                                             name=f'{name}_enc_projection')

        self.enc_blocks = [
            EncoderBlock(enc_heads, enc_dim, enc_mlp_units, dropout, activation=activation,
                         norm=norm, name=f'{name}_enc_block_{i}')
            for i in range(enc_layers)
        ]

        self.enc_norm = norm(name=f'{name}_enc_norm')

        self.dec_projection = Dense(dec_dim, name=f'{name}_dec_projection')

        self.dec_blocks = [
            EncoderBlock(num_heads=dec_heads, mlp_units=dec_mlp_units, dim=dec_dim, dropout=dropout,
                         activation=activation, norm=norm, name=f"{name}_dec_block_{i}")
            for i in range(dec_layers)
        ]

        self.dec_norm = norm(name=f'{name}_dec_norm')

        self.output_projection = Dense(output_patch_height * output_patch_width, name=f'{name}_output_projection')

        self.depatchify = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches,
                                       name=f"{name}_depatchify")

    def call(self, inputs, training=None, mask=None):
        patches = self.patches(inputs)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmasked patches to the encoder.
        encoder_outputs = unmasked_embeddings

        for enc_block in self.enc_blocks:
            encoder_outputs = enc_block(encoder_outputs)

        encoder_outputs = self.enc_norm(encoder_outputs)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # decoder projection
        decoder_outputs = self.dec_projection(decoder_inputs)
        for dec_block in self.dec_blocks:
            decoder_outputs = dec_block(decoder_outputs)

        # output projection
        decoder_outputs = self.output_projection(decoder_outputs)

        # reshape
        decoder_outputs = self.depatchify(decoder_outputs)

        return decoder_outputs, tf.concat([unmask_indices, mask_indices], axis=-1)

    def train_step(self, data):
        x, y = data

        if self.radon:
            # apply radon transform
            sinogram = self.radon_transform(tf.image.resize(tf.expand_dims(y, axis=-1), (362, 362)), training=False)
            sinogram = tf.clip_by_value(sinogram, 0, 10) * 0.46451485

            # preprocess data
            sinogram = add_noise(sinogram, dose=self.dose)
            sinogram, y = preprocess_data(sinogram[:, ::-1, ::-1, 0], y)
        else:
            # preprocess data
            sinogram, y = preprocess_data(x, y, resize_img=False)

        with tf.GradientTape() as tape:
            y_patches = self.patches_2(y)
            y_pred, indices = self(sinogram, training=True)

            y_patches = tf.gather(y_patches, indices, axis=1, batch_dims=1)
            y_patches = self.depatchify(y_patches)

            loss = self.compiled_loss(y_pred, y_patches)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y_pred, y)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        # preprocess data
        sinogram, y = preprocess_data(x, y, resize_img=False)

        # call model
        y_patches = self.patches_2(y)
        y_pred, indices = self(sinogram, training=False)

        y_patches = tf.gather(y_patches, indices, axis=1, batch_dims=1)
        y_patches = self.depatchify(y_patches)

        # evaluate loss
        self.compiled_loss(y_pred, y_patches)
        self.compiled_metrics.update_state(y_pred, y_patches)

        return {m.name: m.result() for m in self.metrics}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        norm_cls = deserialize(config['norm']).__class__
        del config['norm']['config']['name']
        norm = partial(norm_cls, **config['norm']['config'])
        del config['norm']

        return cls(**config, norm=norm)

    def get_config(self):
        cfg = super(MaskedCTransformer, self).get_config()
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
            'radon_transform': self.radon_transform,
            'dose': self.dose
        })
        return cfg


def downstream(mae, num_mask=0, dec_dim=256, dec_layers=8, dec_heads=16, dec_mlp_units=512, output_projection=False,
               output_patch_height=16, output_patch_width=16, output_x_patches=16, output_y_patches=16,
               norm=partial(LayerNormalization, epsilon=1e-5)):
    input_shape = mae.inp_shape
    num_patches = mae.num_patches

    inputs = [Input(input_shape), Input(num_mask, dtype=tf.int32), Input(num_patches - num_mask, dtype=tf.int32)]
    x, mask_indices, unmask_indices = inputs

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

    x = Dense(dec_dim, name='dec_projection')(x)

    for i in range(dec_layers):
        x = EncoderBlock(dec_heads, dec_dim, dec_mlp_units, mae.dropout, activation=mae.activation,
                         norm=norm, name=f'dec_block_{i}')(x)

    x = norm(name='output_norm')(x)
    x = Dense(output_patch_height * output_patch_width, name='output_projection')(x)
    x = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches)(x)

    return Model(inputs=inputs, outputs=x)


def main():
    @tf.function
    def transform(sinogram, gt):
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(8, 1024)), axis=-1
        )
        mask_indices = rand_indices[:, : 768]
        unmask_indices = rand_indices[:, 768:]
        sinogram = tf.expand_dims(sinogram - 42.932495, -1) / 31.87962
        sinogram = tf.image.resize(sinogram, (1024, 513))
        gt = tf.expand_dims(gt - 0.16737686, -1) / 0.11505456
        gt = tf.image.resize(gt, (512, 512))
        return sinogram

    feature_desc = {
        'observation': tf.io.FixedLenFeature([], tf.string),
        'ground_truth': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_example(example_proto):
        res = tf.io.parse_single_example(example_proto, feature_desc)
        observation = tf.io.parse_tensor(res['observation'], out_type=tf.float32)
        ground_truth = tf.io.parse_tensor(res['ground_truth'], out_type=tf.float32)
        observation.set_shape((1000, 513))
        ground_truth.set_shape((362, 362))
        return observation, ground_truth

    train_ds = (tf.data.TFRecordDataset('../lodopab_full_dose_train.tfrecord')
                .map(_parse_example)
                .batch(8)
                .prefetch(tf.data.AUTOTUNE)
                .map(transform))

    test_ds = (tf.data.TFRecordDataset('../lodopab_full_dose_validation.tfrecord')
               .map(_parse_example)
               .batch(8)
               .prefetch(tf.data.AUTOTUNE)
               .map(transform))

    model = MAE(enc_layers=1, dec_layers=1, sinogram_width=513, sinogram_height=1, input_shape=(1024, 513, 1),
                enc_dim=1024, enc_mlp_units=4096, dec_dim=513,
                dec_mlp_units=2048)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mse')
    model.fit(train_ds, epochs=1, steps_per_epoch=1)
    model.save('test_save')


if __name__ == "__main__":
    ct_transformer = MaskedCTransformer()
