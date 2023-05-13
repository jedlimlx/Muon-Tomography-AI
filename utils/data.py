import tensorflow as tf


def preprocess_data(sinogram, gt, resize_img=True, expand_dims=True):
    # some rescaling
    if expand_dims:
        sinogram = tf.expand_dims(sinogram, axis=-1)
    sinogram = (sinogram - 0.030857524) / 0.023017514
    sinogram = tf.image.resize(sinogram, (1024, 513), method="bilinear")

    if expand_dims:
        gt = tf.expand_dims(gt, axis=-1)
    gt = (gt - 0.16737686) / 0.11505456
    if resize_img:
        gt = tf.image.resize(gt, (512, 512))

    return sinogram, gt


def add_noise(img, dose=4096):
    img = dose * tf.math.exp(-img)

    img = img + tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=dose ** 0.5, dtype=tf.float32)
    img = tf.clip_by_value(img / dose, 0.1 / dose, tf.float32.max)
    img = -tf.math.log(img)
    return img
