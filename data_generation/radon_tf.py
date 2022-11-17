import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


@tf.function
def radon_parabeam(img, num_angles, num_detectors):
    """
    Computes the radon transform for a parallel beam configuration
    Args:
        img: Input object
        num_angles: Number of angles (dim 0 of output)
        num_detectors: Number of detectors (dim 1 of output)

    Returns: Output sinogram with dim (batch, num_angles, num_detectors)

    """
    img = tf.transpose(img, (1, 2, 0))
    img_shape = img.shape[0]
    diag_length = np.sqrt(2.0) * img_shape
    detectors = tf.range(-diag_length / 2, diag_length / 2, diag_length / num_detectors)
    lines = tf.reshape((tf.stack(tf.meshgrid(detectors, detectors), axis=-1)), (1, num_detectors * num_detectors, 2, 1))
    lines = tf.cast(lines, tf.float32)
    angles = tf.range(0, np.pi, np.pi / num_angles)
    rot_mat = tf.stack([tf.cos(angles), -tf.sin(angles), tf.sin(angles), tf.cos(angles)], axis=-1)
    rot_mat = tf.reshape(rot_mat, (num_angles, 1, 2, 2))
    lines = tf.reshape(tf.matmul(rot_mat, lines), (-1, 2))
    lines = lines + diag_length / 2

    interp = tfp.math.batch_interp_regular_nd_grid(x=lines, x_ref_min=(0, 0), x_ref_max=(img_shape, img_shape),
                                                   y_ref=img, axis=0, fill_value=0)

    sinogram = tf.reduce_sum(tf.reshape(interp, (num_angles, num_detectors, num_detectors, -1)), axis=2)
    return tf.transpose(sinogram, (2, 0, 1))


def test():
    import matplotlib.pyplot as plt
    t = np.zeros((1, 256, 256))
    t[:, 100:150, 100:150] = 1.0
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    out = radon_parabeam(t, 256, 256)
    plt.imshow(out[0])
    plt.show()


if __name__ == "__main__":
    test()

