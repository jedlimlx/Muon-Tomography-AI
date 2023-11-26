import random
import tensorflow as tf


def random_flip(muons, voxels, extra_length=2):
    horizontal = random.randint(0, 1)
    if horizontal:
        voxels = voxels[:, :, ::-1, :]
        muons = (
                tf.constant([-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1] + [1] * extra_length, dtype=tf.float32) * muons +
                tf.constant([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] + [0] * extra_length, dtype=tf.float32)
        )

    vertical = random.randint(0, 1)
    if vertical:
        voxels = voxels[:, ::-1, :, :]
        muons = (
                tf.constant([1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1] + [1] * extra_length, dtype=tf.float32) * muons +
                tf.constant([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] + [0] * extra_length, dtype=tf.float32)
        )

    return muons, voxels


def random_rotate(muons, voxels):
    if random.randint(0, 1):
        voxels = tf.transpose(voxels, (0, 2, 1, 3))
        muons = tf.concat(
            [
                muons[..., :2][::-1],
                muons[..., 2:3],
                muons[..., 3:5][::-1],
                muons[..., 5:6],
                muons[..., 6:8][::-1],
                muons[..., 8:9],
                muons[..., 9:11][::-1],
                muons[..., 11:12],
                muons[..., 12:],
            ], axis=-1
        )

    return muons, voxels
