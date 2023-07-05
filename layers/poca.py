import tensorflow as tf


def poca(x, p, ver_x, ver_p):
    v = tf.linalg.cross(p, ver_p)

    m = tf.stack([p, -ver_p, v], axis=-1)
    b = ver_x - x

    t = tf.linalg.solve(m, b)
    t, ver_t, _ = tf.unstack(t, axis=-1)
    t = t[..., tf.newaxis]
    ver_t = ver_t[..., tf.newaxis]

    return (x + p * t + ver_x + ver_t * ver_p) / 2
