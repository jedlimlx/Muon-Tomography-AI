import math
import numpy as np
import tensorflow as tf

keras = tf.keras

from keras.models import *
from keras.layers import *
from functools import partial

from tqdm import tqdm

from layers.vision_transformer import Patches, PatchEncoder, DecoderBlock, PatchDecoder, MLP, positional_encoding


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
    norm=partial(LayerNormalization, epsilon=1e-5), timestep_embedding="mlp",
    conditioning="cross-attention", covariance="learned", prediction="noise", model=None
):
    input_shape = mae.inp_shape

    num_patches = mae.num_patches

    inputs = [
        Input(input_shape), Input(num_mask, dtype=tf.int32),
        Input(num_patches, dtype=tf.int32),
        Input((output_patch_height * output_y_patches, output_patch_width * output_x_patches, 1), dtype='float32'),
        Input((1,), dtype=tf.float32),
    ]

    x, mask_indices, unmask_indices, y, tt = inputs
    x_t = y

    if timestep_embedding == "sin-cos":
        time_token = TimeEmbedding(dec_dim)(tt[:, 0])
    else:
        time_token = Dense(dec_dim)(Dense(100, activation="gelu")(tt))

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
    if conditioning == "adaln" or conditioning == "adaln-zero":
        x = concatenate(
            [
                GlobalAveragePooling1D()(x),
                TimeEmbedding(input_shape[1])(tt[:, 0])
            ], axis=1
        )
    elif conditioning == "cross-attention":
        x = concatenate(
            [
                x,
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
    y = Dense(output_patch_height * output_patch_width * (2 if covariance == "learned" else 1),
              name='output_projection')(y)
    y = PatchDecoder(output_patch_width, output_patch_height, output_x_patches, output_y_patches,
                     ignore_last=True, channels=2 if covariance == "learned" else 1)(y)

    if covariance == "learned":
        pred_noise, pred_variance = tf.split(y, 2, axis=-1)
    else: pred_noise = y

    if prediction == "gt":
        pred_noise = (x_t - Extract()([model.alphas, tt, tf.shape(x_t)]) ** 0.5 * pred_noise) / \
                     ((1 - Extract()([model.alphas_cumprod, tt, tf.shape(x_t)])) ** 0.5 /
                      (1 - Extract()([model.alphas, tt, tf.shape(x_t)])))

    if covariance == "learned": y = concatenate([pred_noise, pred_variance])
    else: y = pred_noise

    return Model(inputs=inputs, outputs=y)


class Extract(Layer):
    def call(self, inputs, *args, **kwargs):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        a, t, x_shape = inputs
        t = tf.cast(t, tf.int32)

        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])


class DiffusionModel(Model):
    """
    Implements a diffusion model for solving linear inverse problems
    """

    def __init__(self, model, timesteps=1000, covariance="fixed", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timesteps = timesteps
        self.covariance = covariance

        self.model = model

        self.clip_min = -1
        self.clip_max = 1

        # Define the cosine variance schedule
        self.num_timesteps = int(timesteps)

        s = 0.008
        temp = np.array(
            [np.cos((t / self.num_timesteps + s) / (1 + s) * math.pi / 2) ** 2 for t in range(self.num_timesteps)])
        betas = np.array([0 if x == 0 else min(1 - temp[x] / temp[x - 1], 0.999) for x in range(self.num_timesteps)])

        alphas = 1.0 - betas
        self.alphas = tf.constant(alphas, dtype=tf.float32)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.log_one_minus_alphas_cumprod = tf.constant(
            np.log(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32
        )

        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

        self.loss_tracker = keras.metrics.Mean(name='loss')

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs, *args, **kwargs)

    @tf.function
    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    @tf.function
    def q_mean_variance(self, x_start, t):
        """Extracts the mean, and the variance at current timestep.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return mean, variance, log_variance

    @tf.function
    def q_sample(self, x_start, t, noise):
        """Diffuse the data.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        x_start_shape = tf.shape(x_start)
        return (
                self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start
                + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
                * noise
        )

    @tf.function
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Computes x_0 given x_t, t and the noise
        :param x_t: The image at the current stage in the diffusion process
        :param t: The time of that corresponding image
        :param noise: The noise predicted by the model
        :return: The predicted starting image
        """
        x_t_shape = tf.shape(x_t)
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
                - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    @tf.function
    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """

        x_t_shape = tf.shape(x_t)
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @tf.function
    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        """
        Gets the predicted mean, variance and log variance
        :param pred_noise: The noise predicted by the model
        :param x: Samples at a given timestep for which the noise was predicted
        :param t: Current timestep
        :param clip_denoised: Whether to clip the predicted noise within the specified range
        :return: Returns the mean, variance and log variance
        """
        if self.covariance == "learned":
            pred_noise, pred_variance = tf.split(pred_noise, 2, axis=-1)

            min_log = self._extract(
                self.posterior_log_variance_clipped, t, tf.shape(x)
            )
            max_log = self._extract(np.log(self.betas), t, tf.shape(x))

            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (pred_variance + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = tf.math.exp(model_log_variance)

            x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
            if clip_denoised:
                x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

            model_mean, _, _ = self.q_posterior(
                x_start=x_recon, x_t=x, t=t
            )
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
            if clip_denoised:
                x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

            model_mean, model_variance, model_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t
            )

        return model_mean, model_variance, model_log_variance

    @tf.function
    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffusion model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)

        # No noise when t == 0
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1]
        )
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise

    def diffusion_loss(self, tt, noise_true, noise_pred):
        if self.covariance == "learned":
            noise_pred, covariance = tf.split(noise_pred, 2, axis=-1)

            min_log = self._extract(self.posterior_log_variance_clipped, tt, tf.shape(noise_true))
            max_log = self._extract(np.log(self.betas), tt, tf.shape(noise_true))

            frac = (covariance + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = tf.math.exp(model_log_variance)

            l_simple = tf.reduce_mean(tf.square(noise_true - noise_pred), axis=-1)  # L_simple
            l_vlb = (1 - self._extract(self.alphas, tt, tf.shape(noise_pred))) ** 2 / \
                    (2 * self._extract(self.alphas, tt, tf.shape(noise_pred)) *
                     (1 - self._extract(self.alphas_cumprod, tt, tf.shape(noise_pred))) *
                     tf.norm(model_variance, axis=-1) + 1e-8) * \
                    tf.stop_gradient(l_simple)  # L_vlb
            return l_simple + 0.001 * l_vlb
        else:
            return tf.reduce_mean(tf.square(noise_true - noise_pred), axis=-1)  # L_simple

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        t = tf.random.uniform((tf.shape(y)[0], 1,), minval=1, maxval=self.timesteps)
        t = tf.cast(tf.math.floor(t), tf.int32)
        noise = tf.random.normal((tf.shape(y)[0], 512, 512, 1))
        inputt = self.q_sample(y, t, noise)

        with tf.GradientTape() as tape:
            y_pred = self((*x, inputt, t), training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.diffusion_loss(t, noise, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(noise, y_pred)
        self.loss_tracker.update_state(loss)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def stochastic_sample(self, sinogram, mask_indices, unmask_indices):
        """
        Generates a stochastic sample from the diffusion model
        :param sinogram: The sinogram y that the model uses as a condition
        :param mask_indices: The indices of the sinogram to mask
        :param unmask_indices: The indices of the sinogram to keep unmasked
        :return: Returns the generated samples
        """
        sample_lst = []
        noise_lst = []

        num_images = tf.shape(sinogram)[0]

        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, 512, 512, 1), dtype=tf.float32
        )

        # 2. Sample from the model iteratively
        for t in tqdm(list(reversed(range(1, self.num_timesteps)))):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.model.predict(
                [sinogram, mask_indices, unmask_indices, samples, tf.expand_dims(tt, axis=-1)],
                verbose=0
            )
            samples = self.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
            sample_lst.append(samples)
            noise_lst.append(pred_noise)

        # 3. Return generated samples
        return sample_lst, noise_lst

    @property
    def metrics(self):
        return [*super(DiffusionModel, self).metrics, self.loss_tracker]

    # def test_stochastic_sample(self, sinogram, mask_indices, unmask_indices, gt, start_time):
    #     """
    #     Generates a stochastic sample from the diffusion model starting from a timestep t
    #     :param sinogram: The sinogram y that the model uses as a condition
    #     :param mask_indices: The indices of the sinogram to mask
    #     :param unmask_indices: The indices of the sinogram to keep unmasked
    #     :param gt: The noised image at the timestep t
    #     :param start_time: The timestep t to start from
    #     :return: Returns the generated samples
    #     """
    #     sample_lst = []
    #     noise_lst = []
    #
    #     num_images = tf.shape(sinogram)[0]
    #
    #     # 1. Randomly sample noise (starting point for reverse process)
    #     t = tf.ones(shape=(len(sinogram), 1,)) * start_time
    #     t = tf.cast(t, tf.int32)
    #
    #     noise = tf.random.normal((len(sinogram), 512, 512, 1))
    #     samples = self.q_sample(gt, t, noise)
    #     samples_original = samples
    #
    #     # 2. Sample from the model iteratively
    #     for t in tqdm(list(reversed(range(1, start_time)))):
    #         tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
    #         pred_noise = self.predict(
    #             [sinogram, mask_indices, unmask_indices, samples, tf.expand_dims(tt, axis=-1)],
    #             verbose=0
    #         )
    #         samples = self.p_sample(
    #             pred_noise, samples, tt, clip_denoised=True
    #         )
    #         sample_lst.append(samples)
    #         noise_lst.append(pred_noise)
    #
    #     # 3. Return generated samples
    #     return sample_lst, noise_lst, samples_original


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    import tensorflow as tf
    import tensorflow_addons as tfa

    from tensorflow.keras.layers import *

    from layers.vision_transformer import positional_encoding
    from layers.mae import MAE

    PER_REPLICA_BATCH_SIZE = 1
    # %%
    import math

    interpolation = "bilinear"


    @tf.function
    def transform_mae(sinogram, gt):
        sinogram = tf.expand_dims(sinogram - 0.030857524, -1) / 0.023017514
        sinogram = tf.image.resize(sinogram, (1024, 513))
        return sinogram


    @tf.function
    def transform_denoise(sinogram, gt):
        sinogram = tf.expand_dims(sinogram - 0.030857524, -1) / 0.023017514
        sinogram = tf.image.resize(sinogram, (1024, 513), method=interpolation)
        gt = tf.expand_dims(gt - 0.16737686, -1) / 0.11505456
        gt = tf.image.resize(gt, (512, 512))

        rand_indices = tf.argsort(
            tf.random.uniform(shape=(tf.shape(gt)[0], 1024)), axis=-1
        )
        mask_indices = rand_indices[:, : 0]
        unmask_indices = rand_indices[:, 0:]

        return (sinogram, mask_indices, unmask_indices), gt


    # %%
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


    # %%
    train_ds_denoise = tf.data.TFRecordDataset('lodopab_full_dose_train.tfrecord').map(_parse_example).batch(
        PER_REPLICA_BATCH_SIZE).map(transform_denoise).shuffle(100)

    val_ds_denoise = tf.data.TFRecordDataset('lodopab_full_dose_validation.tfrecord').map(_parse_example).batch(
        PER_REPLICA_BATCH_SIZE).map(transform_denoise).shuffle(100)

    test_ds_denoise = tf.data.TFRecordDataset('lodopab_full_dose_test.tfrecord').map(_parse_example).batch(
        PER_REPLICA_BATCH_SIZE).map(transform_denoise).shuffle(100)

    # train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    # test_dist_ds = strategy.experimental_distribute_dataset(test_ds)
    # %%
    mae = MAE(
        enc_layers=1,
        dec_layers=1,
        sinogram_width=513,
        sinogram_height=1,
        input_shape=(1024, 513, 1),
        enc_dim=512,
        enc_mlp_units=256,
        dec_dim=256,
        dec_mlp_units=2048
    )
    mae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    # history = model.fit(train_ds, epochs=50, validation_data=test_ds)
    # mae.load_weights("../input/lodopab-mae/model_ckpt/model")

    base_model = DiffusionTransformer(
        mae,
        num_mask=0,
        dec_dim=256,
        dec_mlp_units=256,
        dec_layers=1,
        output_patch_width=16,
        output_patch_height=16,
        output_x_patches=32,
        output_y_patches=32,
        timestep_embedding="mlp",
        conditioning="adaln-zero",
        covariance="fixed"
    )

    mae.trainable = True
    base_model.compile(
        optimizer=(
            tfa.optimizers.AdamW(weight_decay=3e-7, learning_rate=1.2e-4, beta_1=0.9, beta_2=0.999)
        ), loss="mse"
    )
    base_model.summary()

    diffusion_model = DiffusionModel(model=base_model)

    diffusion_model.model = base_model
    diffusion_model.compile(
        optimizer=(
            tfa.optimizers.AdamW(weight_decay=3e-7, learning_rate=1.2e-4, beta_1=0.9, beta_2=0.999)
        ), loss=lambda x, y, z: diffusion_model.diffusion_loss(x, y, z)
    )

    history = diffusion_model.fit(train_ds_denoise, epochs=1)
    base_model.save_weights('diffusion_model/model')
