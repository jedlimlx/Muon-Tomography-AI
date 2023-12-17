import numpy as np
import tensorflow as tf


def poca_points(x, p, ver_x, ver_p):
    v = tf.linalg.cross(p, ver_p)

    m = tf.stack([p, -ver_p, v], axis=-1)
    b = ver_x - x

    m_inv = tf.linalg.pinv(m)
    t = tf.linalg.matmul(m_inv, b[..., tf.newaxis])
    scattered = (tf.linalg.det(m) > 1e-8) | (tf.linalg.det(m) < -1e-8)
    not_scattered = tf.cast(~scattered, tf.float32)[..., tf.newaxis]
    scattered = tf.cast(scattered, tf.float32)[..., tf.newaxis]
    t, ver_t, _ = tf.unstack(tf.squeeze(t, axis=-1), axis=-1)
    t = t[..., tf.newaxis]
    ver_t = ver_t[..., tf.newaxis]

    poca_points = (x + p * t + ver_x + ver_t * ver_p) / 2
    poca_points = poca_points * scattered + (x + ver_x) / 2 * not_scattered

    return poca_points, scattered


def poca(x, p, ver_x, ver_p, p_estimate=None, resolution=64, r=1, r2=1):
    b = x.shape[0]
    dosage = x.shape[1] // 256

    coordinates = tf.stack(tf.meshgrid(
        tf.range(resolution, dtype=tf.int32),
        tf.range(resolution, dtype=tf.int32),
        tf.range(resolution, dtype=tf.int32),
    ), axis=-1) / resolution
    coordinates = tf.repeat(tf.repeat(coordinates[tf.newaxis, tf.newaxis, ...], dosage, axis=1), b, axis=0)
    coordinates = tf.cast(coordinates, tf.float32)

    def func(inputs):
        inputs = tf.split(inputs, 4, axis=0)
        x, p, ver_x, ver_p = inputs[0], inputs[1], inputs[2], inputs[3]

        # run poca algorithm to find the scattering points
        scattering, mask = poca_points(x, p / tf.norm(p, axis=-1, keepdims=True), ver_x, ver_p)

        # constructing voxels
        count = tf.zeros((b, resolution, resolution, resolution), dtype=tf.int32)
        
        x_expanded = tf.repeat(
            tf.repeat(
                tf.repeat(
                    x[:, :, tf.newaxis, tf.newaxis, tf.newaxis, ...], resolution, axis=4
                ), resolution, axis=3
            ), resolution, axis=2
        )

        # expanding dimensions (memory going RIP)
        ver_x_expanded = tf.repeat(
            tf.repeat(
                tf.repeat(
                    ver_x[:, :, tf.newaxis, tf.newaxis, tf.newaxis, ...], resolution, axis=4
                ), resolution, axis=3
            ), resolution, axis=2
        )

        scattering_expanded = tf.repeat(
            tf.repeat(
                tf.repeat(
                    scattering[:, :, tf.newaxis, tf.newaxis, tf.newaxis, ...], resolution, axis=4
                ), resolution, axis=3
            ), resolution, axis=2
        )

        # considering first segment
        distance1 = tf.einsum(
            "bdxyz,bd->bdxyz",
            tf.norm(
                tf.linalg.cross(coordinates - ver_x_expanded, scattering_expanded - ver_x_expanded), axis=-1
            ), 1 / tf.norm(scattering - ver_x, axis=-1)
        )

        # considering 2nd segment
        distance2 = tf.einsum(
            "bdxyz,bd->bdxyz",
            tf.norm(
                tf.linalg.cross(coordinates - scattering_expanded, x_expanded - scattering_expanded), axis=-1
            ), 1 / tf.norm(x - scattering, axis=-1)
        )

        # using bitwise OR prevents double counting
        count += tf.math.reduce_sum(
            tf.cast(
                ((distance1 < r2 / resolution) & (coordinates[..., -1] > scattering_expanded[..., -1])) |
                ((distance2 < r2 / resolution) & (coordinates[..., -1] < scattering_expanded[..., -1])), tf.int32
            ), axis=1
        )

        # compute the list of scattering angles
        scattering_angles = tf.math.acos(
            tf.abs(tf.einsum("ijk,ijk->ij", scattering - ver_x, x - scattering)) /
            (tf.norm(scattering - ver_x, axis=-1) * tf.norm(x - scattering, axis=-1))
        )

        # some are nan idk why
        value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(scattering_angles)), dtype=tf.float32)
        scattering_angles = tf.math.multiply_no_nan(scattering_angles, value_not_nan)

        scattering_voxels = tf.einsum(
            "bdxyz,bd->bdxyz",
            tf.math.divide_no_nan(
                tf.cast(
                    tf.math.reduce_sum(
                        tf.square(coordinates - scattering_expanded), axis=-1
                    ) < (r / resolution)**2, tf.float32
                ), (2 * tf.norm(scattering_expanded - ver_x_expanded, axis=-1) * 100)
            ), scattering_angles ** 2 * mask[..., 0] * (tf.norm(p, axis=-1) ** 2 / 15 ** 2)
        )

        return tf.concat([tf.math.reduce_sum(scattering_voxels, axis=1), tf.cast(count, tf.float32)], axis=0)

    x_split = tf.split(x, 256, axis=1)
    p_split = tf.split(p * p_estimate[..., tf.newaxis], 256, axis=1)
    ver_x_split = tf.split(ver_x, 256, axis=1)
    ver_p_split = tf.split(ver_p, 256, axis=1)

    output = tf.map_fn(
        func,
        tf.concat([x_split, p_split, ver_x_split, ver_p_split], axis=1),
        fn_output_signature=tf.float32
    )
    output = tf.split(output, 2, axis=1)
    scattering_voxels, count = output[0], output[1]

    return tf.math.divide_no_nan(tf.math.reduce_sum(scattering_voxels, axis=0), tf.math.reduce_sum(count, axis=0))


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import tensorflow as tf

    # List of materials by radiation length
    # Materials (Air, Concrete, Al, Ti, Fe, Ni, Sn, Ag, Pb, Au, U)
    radiation_lengths = [36.62, 11.55, 8.897, 3.560, 1.757, 1.424, 1.206, 0.8543, 0.5612, 0.3344, 0.3166]
    inverse_radiation_length = [1 / x for x in radiation_lengths]

    # Create a description of the features.
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string)
    }


    def _parse_example(example_proto):
        res = tf.io.parse_single_example(example_proto, feature_description)
        x = tf.io.parse_tensor(res['x'], out_type=tf.double)
        y = tf.io.parse_tensor(res['y'], out_type=tf.int32)
        y.set_shape((64, 64, 64))

        x = tf.cast(x, dtype=tf.float32)
        return x, y


    def set_dosage(x, y, dosage):
        x = x[:dosage]
        x.set_shape((dosage, 12))
        return x, y


    def construct_ds(dosage, p_error=0.2):
        return (
            tf.data.TFRecordDataset("../../Datasets/voxels_prediction_test.tfrecord")
            .map(_parse_example)
            .filter(lambda x, y: len(x) >= dosage)
            .map(lambda x, y: set_dosage(x, y, dosage))
            .map(
                lambda x, y: (
                    tf.concat([
                        x[:, :3] / 1000 + 0.5,
                        # tf.cast(tf.math.rint(x[:, :3] / 1000 * RESOLUTION), tf.float32) / RESOLUTION + 0.5,
                        x[:, 3:6] / tf.norm(x[:, 3:6], axis=-1, keepdims=True),
                        x[:, 6:9] / 1000 + 0.5,
                        # tf.cast(tf.math.rint(x[:, 6:9] / 1000 * RESOLUTION), tf.float32) / RESOLUTION + 0.5,
                        x[:, 9:12],
                        tf.norm(x[:, 3:6], axis=-1, keepdims=True) * tf.random.normal((1,), 1, p_error)
                    ], axis=1), tf.gather_nd(inverse_radiation_length, tf.cast(y[..., tf.newaxis], tf.int32))
                )
            )
            .batch(1)
        )

    ds = construct_ds(16384)

    for x, y in ds.skip(2): break

    x = x.numpy()
    print(x[:, :, -1])

    output, count = poca(
        x[:, :, :3], x[:, :, 3:6], x[:, :, 6:9], x[:, :, 9:12], x[:, :, -1],
        resolution=64, r=5, r2=2
    )
    output = tf.transpose(output, (0, 2, 1, 3))
    # print(output)
