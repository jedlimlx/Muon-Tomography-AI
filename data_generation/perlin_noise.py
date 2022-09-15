import numpy as np
import SimpleITK as itk

"""
Reference: https://pvigier.github.io/2018/11/02/3d-perlin-noise-numpy.html
"""


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
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1], 0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    phi = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    gradients = np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=3)
    gradients[-1] = gradients[0]
    g000 = gradients[0:-1, 0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:, 0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1, 1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:, 1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1, 0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:, 0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1, 1:, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:, 1:, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n100
    n10 = n010 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n110
    n01 = n001 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n101
    n11 = n011 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n111
    n0 = (1 - t[:, :, :, 1]) * n00 + t[:, :, :, 1] * n10
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
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    scaling = 0
    for _ in range(octaves):
        scaling += amplitude
        noise += amplitude * generate_perlin_noise_3d(shape,
                                                      (frequency * res[0], frequency * res[1], frequency * res[2]))
        frequency *= 2
        amplitude *= persistence
    return noise * 0.05 / scaling


def export_to_mhd(filename: str, a: np.ndarray) -> None:
    """
    Exports array to mhd file
    Args:
        filename: filename to save array to
        a: array to save

    Returns:
        None
    """
    img = itk.GetImageFromArray(a)
    itk.WriteImage(img, filename)
