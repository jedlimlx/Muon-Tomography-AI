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
                tf.constant([1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1] + [1] * extra_length, dtype=tf.float32) * muons +
                tf.constant([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] + [0] * extra_length, dtype=tf.float32)
        )

    return muons, voxels


def random_rotate(muons, voxels):
    if random.randint(0, 1):
        voxels = tf.transpose(voxels, (0, 2, 1, 3))
        muons = tf.concat(
            [
                muons[..., 1:2],
                muons[..., 0:1],
                muons[..., 2:3],
                muons[..., 4:5],
                muons[..., 3:4],
                muons[..., 5:6],
                muons[..., 7:8],
                muons[..., 6:7],
                muons[..., 8:9],
                muons[..., 10:11],
                muons[..., 9:10],
                muons[..., 11:]
            ], axis=-1
        )

    return muons, voxels
