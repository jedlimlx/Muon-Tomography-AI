import math
import numpy as np
import tensorflow as tf

keras = tf.keras

from keras.models import *
from keras.layers import *
from functools import partial

from tqdm import tqdm

from layers.vision_transformer import Patches, PatchEncoder, DecoderBlock, PatchDecoder, MLP
from layers.mae import MAE

from utils import preprocess_data, add_noise


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
    conditioning="cross-attention", covariance="learned"
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
                TimeEmbedding(x.shape[-1])(tt)
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

    return Model(inputs=inputs, outputs=y)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tf.math.reduce_mean(tensor, axis=range(1, len(tensor.shape)))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + tf.math.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * tf.math.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + tf.math.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = tf.math.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = tf.math.log(tf.clip_by_value(cdf_plus, 1e-12, 1e+12))
    log_one_minus_cdf_min = tf.math.log(tf.clip_by_value(1.0 - cdf_min, 1e-12, 1e+12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = tf.where(
        x < -0.999,
        log_cdf_plus,
        tf.where(x > 0.999, log_one_minus_cdf_min, tf.math.log(tf.clip_by_value(cdf_delta, 1e-12, 1e+12))),
    )

    assert log_probs.shape == x.shape
    return log_probs


class DiffusionModel(Model):
    """
    Implements a diffusion model for solving linear inverse problems
    """

    def __init__(
        self, model: Model, radon_transform: Model,
        timesteps=1000, covariance="fixed", prediction="noise",
        vlb_weight=1e-5, radon=False, dose=4096, gamma=0.05, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.timesteps = timesteps
        self.covariance = covariance
        self.prediction = prediction
        self.vlb_weight = vlb_weight

        self.radon = radon
        self.radon_transform = radon_transform

        self.dose = dose
        self.gamma = gamma

        self.min_timestep = 0
        self.max_timestep = timesteps - 1

        self.model = model

        self.clip_min = -1
        self.clip_max = 1

        # Define the cosine variance schedule
        self.num_timesteps = int(timesteps)

        s = 0.008
        temp = np.array(
            [np.cos((t / self.num_timesteps + s) / (1 + s) * math.pi / 2) ** 2 for t in range(self.num_timesteps)])
        betas = np.array([1e-8 if x == 0 else min(1 - temp[x] / temp[x - 1], 0.999) for x in range(self.num_timesteps)])

        alphas = 1.0 - betas
        self.alphas = tf.constant(alphas, dtype=tf.float32)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.log_betas = tf.cast(tf.math.log(betas), dtype=tf.float32)
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

        # Weight loss by how important the model's output is at that time-step
        self.weighting = (self.alphas_cumprod_prev / self.alphas_cumprod * (1 - self.alphas_cumprod)) ** 0.5

        self.l_simple_tracker = keras.metrics.Mean(name='l_simple')
        self.l_vlb_tracker = keras.metrics.Mean(name='l_vlb')

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
            max_log = self._extract(self.log_betas, t, tf.shape(x))

            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (pred_variance + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = tf.math.exp(model_log_variance)

            if self.prediction == "noise":
                x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
                if clip_denoised:
                    x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)
            else:
                x_recon = pred_noise

            model_mean, _, _ = self.q_posterior(
                x_start=x_recon, x_t=x, t=t
            )
        else:
            if self.prediction == "noise":
                x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
                if clip_denoised:
                    x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)
            else:
                x_recon = pred_noise

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

    @tf.function
    def ddim_sample(self, pred_noise, x, t, new_t=None, clip_denoised=True, eta=0):
        if self.prediction == "noise": raise NotImplementedError
        if new_t is None: new_t = t - 1

        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )

        eps = (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - pred_noise
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)

        alpha_bar = self._extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod, new_t, x.shape)
        sigma = (
            eta
            * tf.math.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * tf.math.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Equation 12.
        noise = tf.random.normal(tf.shape(x))
        mean_pred = (
            pred_noise * tf.math.sqrt(alpha_bar_prev)
            + tf.math.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1]
        )  # no noise when t == 0

        return mean_pred + nonzero_mask * sigma * noise

    def diffusion_loss(self, t, x_start, x_t, noise_true, model_out):
        if self.covariance == "learned":
            noise_pred, covariance = tf.split(model_out, 2, axis=-1)
            if self.prediction == "gt":
                gt_pred = noise_pred
                l_simple = tf.reduce_mean(tf.square(x_start - gt_pred), axis=-1)  # L_simple
            else:
                l_simple = tf.reduce_mean(tf.square(noise_true - noise_pred), axis=-1)  # L_simple

            # computing the variational lower bound
            t = t[:, 0]

            true_mean, _, true_log_variance_clipped = self.q_posterior(
                x_start=x_start, x_t=x_t, t=t
            )
            mean, variance, log_variance = self.p_mean_variance(model_out, x_t, t)

            kl = normal_kl(true_mean, true_log_variance_clipped, mean, log_variance)
            kl = mean_flat(kl) / np.log(2.0)

            decoder_nll = -discretized_gaussian_log_likelihood(
                x_start, means=mean, log_scales=0.5 * log_variance
            )
            decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

            # At the first timestep return the decoder NLL,
            # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            l_vlb = tf.where(t == 0, decoder_nll, kl)
            return l_simple, tf.reduce_mean(l_vlb)
        else:
            if self.prediction == "noise":
                x_start_pred = self.predict_start_from_noise(x_t, t, model_out)
                return tf.reduce_mean(tf.square(noise_true - model_out), axis=-1) + \
                       0.01 * tf.reduce_mean(tf.square(x_start - x_start_pred), axis=-1), 0
            else:
                return tf.reduce_mean(tf.square(x_start - model_out), axis=-1), 0

    def train_step(self, data):
        x, y = data

        if self.radon:
            # apply radon transform
            sinogram = self.radon_transform(tf.image.resize(tf.expand_dims(y, axis=-1), (362, 362)), training=False)
            sinogram = tf.clip_by_value(sinogram, 0, 10) * 0.46451485

            # preprocess data
            sinogram = add_noise(sinogram, dose=self.dose)
            sinogram, y = preprocess_data(sinogram[:, ::-1, ::-1, -1], y)
            x = (sinogram, x[1], x[2])
        else:
            # preprocess data
            sinogram, y = preprocess_data(x[0], y)
            x = (sinogram, x[1], x[2])

        # generating timesteps
        t = tf.random.uniform((tf.shape(y)[0], 1,), minval=self.min_timestep, maxval=self.max_timestep)
        t = tf.cast(tf.math.floor(t), tf.int32)

        # adding noise to the gt
        noise = tf.random.normal((tf.shape(y)[0], 512, 512, 1))

        # additional noise added to simulate error in predictions
        y_noised = tf.multiply(
            tf.random.normal((tf.shape(y)[0], 512, 512, 1)),
            tf.random.normal((tf.shape(y)[0], 1, 1, 1), mean=0, stddev=self.gamma)
        ) + y

        inputt = self.q_sample(y_noised, t, noise)

        with tf.GradientTape() as tape:
            y_pred = self.model((*x, inputt, t / self.num_timesteps), training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            l_simple, l_vlb = self.diffusion_loss(t, y, inputt, noise, y_pred)
            loss = l_simple + self.vlb_weight * l_vlb

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(noise, y_pred)
        self.l_simple_tracker.update_state(l_simple)
        self.l_vlb_tracker.update_state(l_vlb)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        # don't apply radon transform, we use real data for testing
        # x[0] = radon(y, training=False)
        # x[0] = tf.clip_by_value(x[0], 0, 10) * 0.46451485

        # preprocess data
        sinogram, y = preprocess_data(x[0], y)
        x = (sinogram, x[1], x[2])

        # generating timesteps
        t = tf.random.uniform((tf.shape(y)[0], 1,), minval=self.min_timestep, maxval=self.max_timestep)
        t = tf.cast(tf.math.floor(t), tf.int32)

        # adding noise to the gt
        noise = tf.random.normal((tf.shape(y)[0], 512, 512, 1))
        inputt = self.q_sample(y, t, noise)

        y_pred = self.model((*x, inputt, t / self.num_timesteps), training=False)  # Forward pass

        # Compute the loss value
        # (the loss function is configured in `compile()`)
        l_simple, l_vlb = self.diffusion_loss(t, y, inputt, noise, y_pred)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(noise, y_pred)
        self.l_simple_tracker.update_state(l_simple)
        self.l_vlb_tracker.update_state(l_vlb)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def stochastic_sample(self, sinogram, mask_indices, unmask_indices):
        """
        Generates a stochastic sample from the diffusion model (DDPM sampling)
        :param sinogram: The sinogram y that the model uses as a condition
        :param mask_indices: The indices of the sinogram to mask
        :param unmask_indices: The indices of the sinogram to keep unmasked
        :return: Returns the generated samples
        """
        sample_lst = []

        num_images = tf.shape(sinogram)[0]

        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, 512, 512, 1), dtype=tf.float32
        )

        # 2. Sample from the model iteratively
        for t in tqdm(list(reversed(range(0, self.num_timesteps)))):
            tt = tf.cast(tf.fill([num_images], t), dtype=tf.int64)
            pred_noise = self.model.predict(
                [sinogram, mask_indices, unmask_indices, samples, tf.expand_dims(tt / self.num_timesteps, axis=-1)],
                verbose=0
            )

            samples = self.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
            sample_lst.append(samples)

        # 3. Return generated samples
        return sample_lst

    def deterministic_sample(self, sinogram, mask_indices, unmask_indices, timesteps=None):
        """
        Generates a deterministic sample from the diffusion model (DDIM sampling)
        :param sinogram: The sinogram y that the model uses as a condition
        :param mask_indices: The indices of the sinogram to mask
        :param unmask_indices: The indices of the sinogram to keep unmasked
        :return: Returns the generated samples
        """
        if timesteps is None: timesteps = list(reversed(range(0, self.num_timesteps)))

        sample_lst = []

        num_images = tf.shape(sinogram)[0]

        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, 512, 512, 1), dtype=tf.float32
        )

        # 2. Sample from the model iteratively
        for i, t in tqdm(enumerate(timesteps)):
            tt = tf.cast(tf.fill([num_images], t), dtype=tf.int64)
            tt_new = tf.cast(tf.fill([num_images], timesteps[i + 1]), dtype=tf.int64)
            pred_noise = self.model.predict(
                [sinogram, mask_indices, unmask_indices, samples, tf.expand_dims(tt / self.num_timesteps, axis=-1)],
                verbose=0
            )

            samples = self.ddim_sample(
                pred_noise, samples, tt, new_t=tt_new, clip_denoised=True
            )
            sample_lst.append(samples)

        # 3. Return generated samples
        return sample_lst

    def test_stochastic_sample(self, sinogram, mask_indices, unmask_indices, gt, start_time):
        sample_lst = []

        num_images = tf.shape(sinogram)[0]

        # 1. Randomly sample noise (starting point for reverse process)
        samples = gt

        # 2. Sample from the model iteratively
        for t in tqdm(list(reversed(range(1, start_time)))):
            tt = tf.cast(tf.fill([num_images], t), dtype=tf.int64)

            pred_noise = self.model.predict(
                [sinogram, mask_indices, unmask_indices, samples, tf.expand_dims(tt / self.num_timesteps, axis=-1)],
                verbose=0
            )

            samples = self.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
            sample_lst.append(samples)

        # 3. Return generated samples
        return sample_lst

    @property
    def metrics(self):
        return [*super(DiffusionModel, self).metrics, self.l_simple_tracker, self.l_vlb_tracker]


if __name__ == "__main__":
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
        output_y_patches=32
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
