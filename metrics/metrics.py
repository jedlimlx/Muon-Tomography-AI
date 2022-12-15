import tensorflow as tf


class SSIM(tf.keras.metrics.Metric):
    def __init__(self, d_range=None, rescaling=False, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.shape = None
        self.d_range = d_range
        self.rescaling = rescaling
        self.mean = mean
        self.std = std

        self.sum = self.add_weight('sum', initializer='zeros')
        self.samples = self.add_weight('samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.rescaling:
            y_true = y_true * self.std + self.mean
            y_pred = y_pred * self.std + self.mean

        if self.d_range is None:
            d_range = tf.reduce_max(y_true, axis=(1, 2), keepdims=True) - \
                      tf.reduce_min(y_true, axis=(1, 2), keepdims=True)
        else:
            d_range = self.d_range

        self.sum.assign_add(tf.reduce_mean(tf.image.ssim(y_true, y_pred, d_range)))
        self.samples.assign_add(1., tf.float32)

    def result(self):
        return self.sum / self.samples

class PSNR(tf.keras.metrics.Metric):
    def __init__(self, d_range=None, rescaling=False, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.shape = None
        self.d_range = d_range
        self.rescaling = rescaling
        self.mean = mean
        self.std = std

        self.sum = self.add_weight('sum', initializer='zeros')
        self.samples = self.add_weight('samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.rescaling:
            y_true = y_true * self.std + self.mean
            y_pred = y_pred * self.std + self.mean

        if self.d_range is None:
            d_range = tf.reshape(tf.reduce_max(y_true, axis=(1, 2)) - tf.reduce_min(y_true, axis=(1, 2)), (-1,))
        else:
            d_range = self.d_range

        self.sum.assign_add(tf.reduce_mean(tf.image.psnr(y_true, y_pred, d_range)))
        self.samples.assign_add(1., tf.float32)

    def result(self):
        return self.sum / self.samples
