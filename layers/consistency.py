import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from layers.vision_transformer import positional_encoding
from keras.engine import data_adapter

from utils import preprocess_data


class ConsistencyModel(Model):
    """
    Just a convenience class to use model.fit.
    It will use the optimizer/loss of base_model
    """

    def __init__(self, base_model: Model, ema_model: Model, inp_width, ema=0.999, n_steps=120, t_min=1e-4, rho=7.,
                 num_inference_steps=1, loss_coeff_decay=0.99, init_loss_coeff=1., *args, **kwargs):
        super(ConsistencyModel, self).__init__(*args, **kwargs)

        self.base_model = base_model
        self.ema_model = ema_model

        self.ema = ema
        self.n_steps = n_steps
        self.n_steps_float = tf.cast(n_steps, self.dtype)
        self.t_min = tf.convert_to_tensor(t_min, dtype=self.dtype)
        self.rho = tf.convert_to_tensor(rho, dtype=self.dtype)
        self.num_inference_steps = num_inference_steps

        self.t_encoding = tf.cast(positional_encoding(n_steps, inp_width), self.dtype)[0]
        self.t_arr = self.sample_t(tf.range(0, self.n_steps, dtype=self.dtype))
        self.loss_coeff = tf.Variable(init_loss_coeff, dtype=self.dtype)
        self.loss_coeff_decay = tf.convert_to_tensor(loss_coeff_decay)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.base_model.compile(*args, **kwargs)

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        sinogram, _a, _b, fbp = x

        std_dev_est = tf.math.sqrt((fbp - y) ** 2)
        noise = tf.random.normal(shape=tf.shape(y), mean=y, stddev=std_dev_est)

        n = tf.random.uniform((), 0, self.n_steps - 1, dtype=tf.int32)
        t_embed = tf.reshape(self.t_encoding[n], (1, 1, -1, 1))
        t_embed = tf.broadcast_to(t_embed,
                                  (tf.shape(sinogram)[0], 1, tf.shape(sinogram)[2], tf.shape(sinogram)[3]))

        sinogram = tf.concat([sinogram, t_embed], axis=1)

        _, y_orig = preprocess_data(x[0], y, resize_img=False, expand_dims=False)
        sinogram_temp, y = preprocess_data(x[0], y, resize_img=True, expand_dims=False)
        _, noise = preprocess_data(x[0], noise, resize_img=True, expand_dims=False)

        y_ema = tf.stop_gradient(self.ema_model((sinogram_temp, _a, _b, y + self.t_arr[n] * noise), training=False))

        with tf.GradientTape() as tape:
            y_pred = self.base_model((sinogram, _a, _b, y + self.t_arr[n + 1] * noise), training=True)
            loss = self.base_model.compute_loss(x, y_ema, y_pred, sample_weight)
            loss += self.base_model.compute_loss(x, y_orig, y_pred, sample_weight)

        # tf.print("\n", tf.math.reduce_std(y_pred), tf.math.reduce_std(y_ema))

        gradients = tape.gradient(loss, self.base_model.trainable_variables)
        self.base_model.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))

        for weight, ema_weight in zip(self.base_model.weights, self.ema_model.weights):
            ema_weight.assign(tf.stop_gradient(self.ema * ema_weight + (1 - self.ema) * weight))
        self.base_model.compiled_metrics.update_state(y_orig, y_pred)

        self.loss_coeff.assign(self.loss_coeff * self.loss_coeff_decay)

        return {m.name: m.result() for m in self.base_model.metrics}

    def predict_step(self, data):
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        return self.generate_output(x)

    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        _, y = preprocess_data(x[0], y, resize_img=False, expand_dims=False)

        y_pred = self.generate_output(x)

        self.base_model.compiled_loss(y, y_pred)
        self.base_model.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.base_model.metrics}

    def generate_output(self, x):
        sinogram, _a, _b, fbp = x
        _, fbp = preprocess_data(sinogram, x[3], resize_img=True, expand_dims=False)
        denoised = fbp

        for n in np.linspace(0, self.n_steps - 1, self.num_inference_steps).astype(int)[::-1]:
            t_embed = tf.reshape(self.t_encoding[n], (1, 1, -1, 1))
            t_embed = tf.broadcast_to(t_embed,
                                      (tf.shape(sinogram)[0], 1, tf.shape(sinogram)[2], tf.shape(sinogram)[3]))
            sinogram = tf.concat([sinogram, t_embed], axis=1)
            sinogram, _ = preprocess_data(sinogram, denoised, resize_img=False, expand_dims=False)

            denoised = self.ema_model((sinogram, _a, _b, denoised), training=False)

            if n != 0:
                denoised = tf.image.resize(denoised, (fbp.shape[1], fbp.shape[2]))
                t = self.sample_t(n)
                std_dev_est = tf.sqrt((fbp - denoised) ** 2)
                noise = tf.random.normal(shape=tf.shape(fbp), mean=denoised, stddev=std_dev_est)

                denoised += t * noise

        return denoised

    def sample_t(self, n):
        n = tf.cast(n, self.dtype)
        rho_inv = 1 / self.rho
        return (self.t_min ** rho_inv + n / (self.n_steps_float - 1) * (
                    1. ** rho_inv - self.t_min ** rho_inv)) ** self.rho

    @property
    def metrics(self):
        return self.base_model.metrics
