import tensorflow as tf
import keras


class SSIM(keras.metrics.Mean):
    def __init__(self, d_range=None, rescaling=False, mean=None, std=None, name='ssim', **kwargs):
        super(SSIM, self).__init__(name=name, **kwargs)
        self.shape = None
        self.d_range = d_range
        self.rescaling = rescaling
        self.mean = mean
        self.std = std

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.rescaling:
            y_true = tf.clip_by_value(y_true * self.std + self.mean, 0., 1.)
            y_pred = tf.clip_by_value(y_pred * self.std + self.mean, 0., 1.)

        if self.d_range is None:
            d_range = tf.reduce_max(y_true, axis=(1, 2), keepdims=True) - \
                      tf.reduce_min(y_true, axis=(1, 2), keepdims=True)
        else:
            d_range = self.d_range

        super(SSIM, self).update_state(tf.image.ssim(y_true, y_pred, d_range))

    def get_config(self):
        cfg = super(SSIM, self).get_config()
        cfg.update({
            'd_range': self.d_range,
            'rescaling': self.rescaling,
            'mean': self.mean,
            'std': self.std
        })
        return cfg


class PSNR(keras.metrics.Mean):
    def __init__(self, d_range=None, rescaling=False, mean=None, std=None, name='psnr', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.d_range = d_range
        self.rescaling = rescaling
        self.mean = mean
        self.std = std

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.rescaling:
            y_true = tf.clip_by_value(y_true * self.std + self.mean, 0., 1.)
            y_pred = tf.clip_by_value(y_pred * self.std + self.mean, 0., 1.)

        if self.d_range is None:
            d_range = tf.reshape(tf.reduce_max(y_true, axis=(1, 2)) - tf.reduce_min(y_true, axis=(1, 2)), (-1,))
        else:
            d_range = self.d_range

        super(PSNR, self).update_state(tf.image.psnr(y_true, y_pred, d_range))

    def get_config(self):
        cfg = super(PSNR, self).get_config()
        cfg.update({
            'd_range': self.d_range,
            'rescaling': self.rescaling,
            'mean': self.mean,
            'std': self.std
        })
        return cfg
