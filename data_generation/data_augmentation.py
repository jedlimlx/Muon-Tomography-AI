import random
import tensorflow as tf


def random_flip(muons, voxels, extra_length=2):
    horizontal = random.randint(0, 1)
    if horizontal:
        voxels = voxels[:, ::-1, :, :]
        muons = (
                tf.constant([-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1] + [1] * extra_length) * muons +
                tf.constant([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] + [0] * extra_length)
        )

    vertical = random.randint(0, 1)
    if vertical:
        voxels = voxels[:, :, ::-1, :]
        muons = (
                tf.constant([1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1] + [1] * extra_length) * muons +
                tf.constant([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] + [0] * extra_length)
        )

    return muons, voxels


def random_rotate(muons, voxels):
    if random.randint(0, 1):
        voxels = tf.transpose(voxels, (0, 2, 1, 3, 4))
        muons = tf.concat(
            [
                muons[..., :2][::-1],
                muons[..., 2:3],
                muons[..., 3:6],
                muons[..., 6:8][::-1],
                muons[..., 8:9],
                muons[..., 9:12],
                muons[..., 12:],
            ], axis=1
        )

    return muons, voxels
