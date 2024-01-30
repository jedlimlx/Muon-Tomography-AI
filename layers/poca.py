import tensorflow as tf


def poca(x, p, ver_x, ver_p, threshold=1e-8, nn=None, not_scattered_mul=None):
    if nn is None:
        v = tf.linalg.cross(p, ver_p)

        m = tf.stack([p, -ver_p, v], axis=-1)
        b = ver_x - x

        m_inv = tf.linalg.pinv(m)
        t = tf.linalg.matmul(m_inv, b[..., tf.newaxis])
        scattered = (tf.linalg.det(m) > threshold) | (tf.linalg.det(m) < -threshold)
        not_scattered = tf.cast(~scattered, tf.float32)[..., tf.newaxis]
        scattered = tf.cast(scattered, tf.float32)[..., tf.newaxis]
        t, ver_t, _ = tf.unstack(tf.squeeze(t, axis=-1), axis=-1)
        t = t[..., tf.newaxis]
        ver_t = ver_t[..., tf.newaxis]

        poca_points = (x + p * t + ver_x + ver_t * ver_p) / 2

        if not_scattered_mul is None:
            k = tf.random.uniform(shape=tf.shape(scattered)[:-1], minval=0, maxval=1)
            poca_points = poca_points * scattered + (
                    k[..., tf.newaxis] * x + (1 - k[..., tf.newaxis]) * ver_x
            ) * not_scattered
        else:
            poca_points = poca_points * scattered + (
                    not_scattered_mul * x + (1 - not_scattered_mul) * ver_x
            ) * not_scattered

        return poca_points
    else:
        output = nn(tf.concat([x, p, ver_x, ver_p], axis=-1), training=False)

        n = tf.math.rint(output[:, :, 0:1])
        mask = (output * 0 + tf.range(16, dtype=tf.float32)) <= 3 * tf.repeat(n, 16, axis=-1)

        masked_output = output * tf.cast(mask, tf.float32)
        return tf.concat([
            masked_output[:, :, 1:4],
            masked_output[:, :, 4:7],
            masked_output[:, :, 7:10],
            masked_output[:, :, 10:13],
            masked_output[:, :, 13:16]
        ], axis=1)
