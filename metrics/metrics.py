import tensorflow as tf


def ssim(d_range=None):
    if d_range is None:
        d_range = lambda x: tf.reduce_max(x, axis=(1, 2), keepdims=True) - tf.reduce_min(x, axis=(1, 2), keepdims=True)
    else:
        d_range = lambda x: d_range

    @tf.function
    def apply(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, d_range(y_true))

    return apply


def psnr(d_range=None):
    if d_range is None:
        # the way ssim and pnsr wants the d_range is slightly different
        d_range = lambda x: tf.squeeze(tf.reduce_max(x, axis=(1, 2)) - tf.reduce_min(x, axis=(1, 2)))
    else:
        d_range = lambda x: d_range

    @tf.function
    def apply(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, d_range(y_true))

    return apply
