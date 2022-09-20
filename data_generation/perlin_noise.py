import tensorflow as tf
import numpy as np


def generate_perlin_noise_3d(shape, res):
    """
    Generates 3D perlin noise image
    Args:
        shape: output shape
        res: resolution of Perlin noise grid

    Returns:
        3D image
    """

    def f(t_):
        return 6 * t_ ** 5 - 15 * t_ ** 4 + 10 * t_ ** 3

    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = tf.meshgrid(tf.range(0, res[0], delta[0]),
                       tf.range(0, res[1], delta[1]),
                       tf.range(0, res[2], delta[2]), indexing="ij")
    grid = tf.stack(grid, axis=-1)
    grid = grid - tf.floor(grid)
    grid = tf.cast(grid, tf.float32)
    t = f(grid)
    # Gradients
    theta = 2 * np.pi * tf.random.uniform(shape=[res[0], res[1] + 1, res[2] + 1])
    phi = 2 * np.pi * tf.random.uniform(shape=[res[0], res[1] + 1, res[2] + 1])
    gradients = tf.stack((tf.sin(phi) * tf.cos(theta), tf.sin(phi) * tf.sin(theta), tf.cos(phi)), axis=3)
    gradients = tf.concat([gradients, tf.expand_dims(gradients[0], 0)], axis=0)

    g000 = tf.repeat(tf.repeat(tf.repeat(gradients[0:-1, 0:-1, 0:-1], d[0], 0), d[1], 1), d[2], 2)
    n000 = tf.reduce_sum(tf.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g000, 3)
    g100 = tf.repeat(tf.repeat(tf.repeat(gradients[1:, 0:-1, 0:-1], d[0], 0), d[1], 1), d[2], 2)
    n100 = tf.reduce_sum(tf.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g100, 3)
    n00 = n000 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n100

    g010 = tf.repeat(tf.repeat(tf.repeat(gradients[0:-1, 1:, 0:-1], d[0], 0), d[1], 1), d[2], 2)
    n010 = tf.reduce_sum(tf.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g010, 3)
    g110 = tf.repeat(tf.repeat(tf.repeat(gradients[1:, 1:, 0:-1], d[0], 0), d[1], 1), d[2], 2)
    n110 = tf.reduce_sum(tf.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g110, 3)
    n10 = n010 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n110

    n0 = (1 - t[:, :, :, 1]) * n00 + t[:, :, :, 1] * n10

    g001 = tf.repeat(tf.repeat(tf.repeat(gradients[0:-1, 0:-1, 1:], d[0], 0), d[1], 1), d[2], 2)
    n001 = tf.reduce_sum(tf.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g001, 3)
    g101 = tf.repeat(tf.repeat(tf.repeat(gradients[1:, 0:-1, 1:], d[0], 0), d[1], 1), d[2], 2)
    n101 = tf.reduce_sum(tf.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g101, 3)
    n01 = n001 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n101

    g011 = tf.repeat(tf.repeat(tf.repeat(gradients[0:-1, 1:, 1:], d[0], 0), d[1], 1), d[2], 2)
    n011 = tf.reduce_sum(tf.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g011, 3)
    g111 = tf.repeat(tf.repeat(tf.repeat(gradients[1:, 1:, 1:], d[0], 0), d[1], 1), d[2], 2)
    n111 = tf.reduce_sum(tf.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g111, 3)
    n11 = n011 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n111

    n1 = (1 - t[:, :, :, 1]) * n01 + t[:, :, :, 1] * n11

    return (1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1 + 0.5


def generate_fractal_noise_3d(shape, res, octaves=1, persistence=0.5):
    """
    Generates Perlin noise of different frequencies
    Args:
        shape: shape of output image
        res: resolution of Perlin noise of the lowest frequency
        octaves: number of different frequencies'
        persistence: amount to decrease amplitude by after frequency increases

    Returns:
        3D image
    """
    noise = tf.zeros(shape=shape)
    frequency = 1
    amplitude = 1.0
    scaling = 0
    for _ in range(octaves):
        scaling += amplitude
        noise += amplitude * generate_perlin_noise_3d(shape,
                                                      (frequency * res[0], frequency * res[1], frequency * res[2]))
        noise = tf.stop_gradient(noise)
        frequency *= 2
        amplitude *= persistence

    return noise * 0.5 / scaling


# testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img = generate_fractal_noise_3d([256, 256, 256], [2, 2, 2], octaves=3)
    print(tf.math.reduce_std(img), tf.math.reduce_mean(img))
    plt.imshow(img[0].numpy())
    plt.show()
