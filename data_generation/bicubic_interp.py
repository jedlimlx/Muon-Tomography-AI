import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


def _hermite(A, B, C, D, t):
    a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
    b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
    c = A * (-0.5) + C * 0.5
    d = B

    return a * t * t * t + b * t * t + c * t + d


# def _get_grid_array(n_i, y_i, x_i, c_i):
#     n, y, x, c = np.meshgrid(n_i, y_i, x_i, c_i, indexing='ij')
#     n = np.expand_dims(n, axis=4)
#     y = np.expand_dims(y, axis=4)
#     x = np.expand_dims(x, axis=4)
#     c = np.expand_dims(c, axis=4)
#
#     return np.concatenate([n, y, x, c], axis=4)


def _get_grid_array(n_i, grid, c_i):
    h, w, _, = grid.shape
    c = c_i.shape[0]
    n = n_i.shape[0]
    n_i = tf.reshape(n_i, (-1, 1, 1, 1, 1))
    n_i = tf.tile(n_i, (1, h, w, c, 1))

    grid = tf.reshape(grid, (1, h, w, 1, 2))
    grid = tf.tile(grid, (n, 1, 1, c, 1))

    c_i = tf.reshape(c_i, (1, 1, 1, -1, 1))
    c_i = tf.tile(c_i, (n, h, w, 1, 1))

    return tf.concat([n_i, grid, c_i], axis=4)


# def _get_frac_array(y_d, x_d, n, c):
#     y = y_d.shape[0]
#     x = x_d.shape[0]
#     y_t = y_d.reshape([1, -1, 1, 1])
#     x_t = x_d.reshape([1, 1, -1, 1])
#     y_t = tf.convert_to_tensor(np.tile(y_t, (n, 1, x, c)), dtype=tf.float32)
#     x_t = tf.convert_to_tensor(np.tile(x_t, (n, y, 1, c)), dtype=tf.float32)
#     return y_t, x_t


def _get_frac_array(grid_d, n, c):
    y, x, _ = grid_d.shape

    grid_d = tf.reshape(grid_d, (1, y, x, 1, 2))
    grid_d = tf.tile(grid_d, (n, 1, 1, c, 1))

    return tf.unstack(grid_d, axis=-1)


def _get_index_tensor(grid, height, width, x, y):
    grid_y = grid[:, :, :, :, 1] + y
    grid_x = grid[:, :, :, :, 2] + x

    grid_y = tf.clip_by_value(grid_y, 0, height - 1)
    grid_x = tf.clip_by_value(grid_x, 0, width - 1)

    grid = tf.stack([grid[..., 0], grid_y, grid_x, grid[..., 3]], axis=-1)

    return tf.convert_to_tensor(grid, dtype=tf.int32)


def bicubic_interp(input_, points):
    """

    Args:
        input_: Input tensor. Its shape should be
          [batch_size, height, width, channel].
          In this implementation, the shape should be fixed for speed.
        points: Grid points to compute interpolation.

    Returns:
        Bicubic interpolation of input_ at sample points.
    """

    shape = input_.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channel = shape[3]

    n_i = np.arange(batch_size)
    c_i = np.arange(channel)

    points = tf.cast(points, input_.dtype)
    grid = tf.cast(points, tf.int32)
    grid_d = points - tf.floor(points)

    grid = _get_grid_array(n_i, grid, c_i)
    y_t, x_t = _get_frac_array(grid_d, batch_size, channel)

    i_00 = _get_index_tensor(grid, height, width, -1, -1)
    i_10 = _get_index_tensor(grid, height, width, +0, -1)
    i_20 = _get_index_tensor(grid, height, width, +1, -1)
    i_30 = _get_index_tensor(grid, height, width, +2, -1)

    i_01 = _get_index_tensor(grid, height, width, -1, +0)
    i_11 = _get_index_tensor(grid, height, width, +0, +0)
    i_21 = _get_index_tensor(grid, height, width, +1, +0)
    i_31 = _get_index_tensor(grid, height, width, +2, +0)

    i_02 = _get_index_tensor(grid, height, width, -1, +1)
    i_12 = _get_index_tensor(grid, height, width, +0, +1)
    i_22 = _get_index_tensor(grid, height, width, +1, +1)
    i_32 = _get_index_tensor(grid, height, width, +2, +1)

    i_03 = _get_index_tensor(grid, height, width, -1, +2)
    i_13 = _get_index_tensor(grid, height, width, +0, +2)
    i_23 = _get_index_tensor(grid, height, width, +1, +2)
    i_33 = _get_index_tensor(grid, height, width, +2, +2)

    p_00 = tf.gather_nd(input_, i_00)
    p_10 = tf.gather_nd(input_, i_10)
    p_20 = tf.gather_nd(input_, i_20)
    p_30 = tf.gather_nd(input_, i_30)

    p_01 = tf.gather_nd(input_, i_01)
    p_11 = tf.gather_nd(input_, i_11)
    p_21 = tf.gather_nd(input_, i_21)
    p_31 = tf.gather_nd(input_, i_31)

    p_02 = tf.gather_nd(input_, i_02)
    p_12 = tf.gather_nd(input_, i_12)
    p_22 = tf.gather_nd(input_, i_22)
    p_32 = tf.gather_nd(input_, i_32)

    p_03 = tf.gather_nd(input_, i_03)
    p_13 = tf.gather_nd(input_, i_13)
    p_23 = tf.gather_nd(input_, i_23)
    p_33 = tf.gather_nd(input_, i_33)

    col0 = _hermite(p_00, p_10, p_20, p_30, x_t)
    col1 = _hermite(p_01, p_11, p_21, p_31, x_t)
    col2 = _hermite(p_02, p_12, p_22, p_32, x_t)
    col3 = _hermite(p_03, p_13, p_23, p_33, x_t)
    value = _hermite(col0, col1, col2, col3, y_t)

    value = value * tf.cast((points[..., :1] >= 0) & (points[..., :1] <= height) &
                            (points[..., 1:] >= 0) & (points[..., 1:] <= width), value.dtype)

    return value


def bicubic_resize(input_, new_size, endpoint=False):
    """
    Args :
      input_ : Input tensor. Its shape should be
          [batch_size, height, width, channel].
          In this implementation, the shape should be fixed for speed.
      new_size : The output size [new_height, new_width]
      endpoint: whether the endpoint is included in the interp
    ref :
      http://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
    """

    shape = input_.shape
    height = shape[1]
    width = shape[2]

    new_height = new_size[0]
    new_width = new_size[1]

    if endpoint:
        y_f = np.linspace(0., height - 1, new_height)
    else:
        y_f = np.linspace(0., height, new_height, endpoint=False)

    if endpoint:
        x_f = np.linspace(0., width - 1, new_width)
    else:
        x_f = np.linspace(0., width, new_width, endpoint=False)

    grid = np.stack(np.meshgrid(y_f, x_f, indexing='ij'), axis=-1)

    return bicubic_interp(input_, grid)


def init_causal_coeff(c):
    zn = tf.convert_to_tensor(np.sqrt(3.0) - 2, dtype=c[0].dtype)

    s = c[0]
    for i in range(0, len(c)):
        s += zn * c[i]
        zn *= tf.convert_to_tensor(np.sqrt(3.0) - 2, dtype=c[0].dtype)

    return s


def prefilter_1d(coeffs, data_length):
    pole = tf.convert_to_tensor(np.sqrt(3.0) - 2, dtype=coeffs[0].dtype)
    lamb = (1. - pole) * (1. - 1. / pole)

    # these two loops are probably very slow. idk how to optimize them
    # causal recursion
    coeffs[0] = lamb * init_causal_coeff(coeffs)
    for i in range(1, data_length):
        coeffs[i] = coeffs[i] * lamb + pole * coeffs[i - 1]

    # anti-causal recursion
    coeffs[data_length - 1] = pole / (pole - 1.) * coeffs[data_length - 1]
    for i in range(data_length - 2, -1, -1):
        coeffs[i] = pole * (coeffs[i + 1] - coeffs[i])


def bicubic_prefilter(df: tf.Tensor):
    n, h, w, c = df.shape

    coeffs_h = tf.unstack(df, axis=1)
    prefilter_1d(coeffs_h, h)
    df = tf.stack(coeffs_h, axis=1)

    coeffs_w = tf.unstack(df, axis=2)
    prefilter_1d(coeffs_w, w)
    df = tf.stack(coeffs_w, axis=2)

    return df


def bicubic_interp1(f, points):
    f = bicubic_prefilter(f)
    index = tf.math.floor(points)
    fraction = points - index

    one_frac = 1. - fraction
    squared = fraction * fraction
    one_sqd = one_frac * one_frac

    w0 = 1.0 / 6.0 * one_sqd * one_frac
    w1 = 2.0 / 3.0 - 0.5 * squared * (2.0 - fraction)
    w2 = 2.0 / 3.0 - 0.5 * one_sqd * (2.0 - one_frac)
    w3 = 1.0 / 6.0 * squared * fraction

    g0 = w0 + w1
    g1 = w2 + w3

    h0 = w1 / g0 - 0.5 + index
    h1 = w3 / g1 + 1.5 + index

    print(tf.reduce_min(h0), tf.reduce_max(h0), tf.reduce_min(h1), tf.reduce_max(h1))

    f00 = tfa.image.interpolate_bilinear(f, h0)
    f10 = tfa.image.interpolate_bilinear(f, tf.stack([h1[..., 0], h0[..., 1]], axis=-1))
    f01 = tfa.image.interpolate_bilinear(f, tf.stack([h0[..., 0], h1[..., 1]], axis=-1))
    f11 = tfa.image.interpolate_bilinear(f, h1)

    f00 = g0[..., 1:] * f00 + g1[..., 1:] * f01
    f10 = g0[..., 1:] * f10 + g1[..., 1:] * f11

    return g0[..., :1] * f00 + g1[..., :1] * f10


if __name__ == "__main__":
    print(bicubic_prefilter(tf.reshape(np.linspace(0, 1, 10), (1, 10, 1, 1))))
