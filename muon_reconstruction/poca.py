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


def poca_scattering_density(x, p, ver_x, ver_p, p_estimate=None, resolution=64, r=1):
    b = x.shape[0]
    dosage = x.shape[1]

    coordinates = np.stack(np.meshgrid(
        np.arange(resolution, dtype=np.int32),
        np.arange(resolution, dtype=np.int32),
        np.arange(resolution, dtype=np.int32),
    ), axis=-1) / resolution
    coordinates = np.repeat(np.repeat(coordinates[np.newaxis, np.newaxis, ...], dosage, axis=1), b, axis=0)

    # constructing voxels
    count = np.zeros((b, resolution, resolution, resolution))

    # run poca algorithm to find the scattering points
    scattering, mask = poca_points(x, p, ver_x, ver_p)

    # expanding dimensions (memory going RIP)
    ver_x_expanded = np.repeat(
        np.repeat(
            np.repeat(
                ver_x[:, :, np.newaxis, np.newaxis, np.newaxis, ...], resolution, axis=4
            ), resolution, axis=3
        ), resolution, axis=2
    )

    x_expanded = np.repeat(
        np.repeat(
            np.repeat(
                x[:, :, np.newaxis, np.newaxis, np.newaxis, ...], resolution, axis=4
            ), resolution, axis=3
        ), resolution, axis=2
    )

    scattering_expanded = np.repeat(
        np.repeat(
            np.repeat(
                scattering[:, :, np.newaxis, np.newaxis, np.newaxis, ...], resolution, axis=4
            ), resolution, axis=3
        ), resolution, axis=2
    )

    # considering first segment
    distance1 = np.linalg.norm(
        np.cross(coordinates - ver_x_expanded, scattering_expanded - ver_x_expanded, axis=-1), axis=-1
    ) / np.linalg.norm(scattering - ver_x)

    # considering 2nd segment
    distance2 = np.linalg.norm(
        np.cross(coordinates - scattering_expanded, x_expanded - scattering_expanded, axis=-1), axis=-1
    ) / np.linalg.norm(x - scattering)

    # using bitwise OR prevents double counting
    count += np.sum(
        np.bitwise_or(
            np.bitwise_and(
                distance1 < (r / resolution)**2,
                coordinates[..., -1] > scattering_expanded[..., -1]
            ),
            np.bitwise_and(
                distance2 < (r / resolution)**2,
                coordinates[..., -1] < scattering_expanded[..., -1]
            ),
        ).astype(np.int32), axis=1
    )

    # compute the list of scattering angles
    scattering_angles = np.arccos(np.einsum("ijk,ijl->ij", scattering - ver_x, x - scattering) /
        (np.linalg.norm(scattering - ver_x) * np.linalg.norm(x - scattering)))
    scattering_voxels = np.einsum(
        "bdxyz,bd->bdxyz",
        (np.sum(np.square(coordinates - scattering_expanded) < (r / resolution)**2, axis=-1) == 3).astype(np.int32),
        scattering_angles * mask[..., 0]
    )

    radiation_length = np.sum(scattering_voxels, axis=1) ** 2 * resolution / 2 * (np.mean(p_estimate) ** 2 / 13.6 ** 2) / count # / 10**6
    return radiation_length


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
            tf.data.TFRecordDataset("../voxels_prediction.tfrecord")
            .map(_parse_example)
            .filter(lambda x, y: len(x) >= dosage)
            .map(lambda x, y: set_dosage(x, y, dosage))
            .map(
                lambda x, y: (
                    tf.concat([
                        x[:, :3] / 1000 + 0.5,
                        # tf.cast(tf.math.rint(x[:, :3] / 1000 * RESOLUTION), tf.float32) / RESOLUTION + 0.5,
                        x[:, 3:6] / tf.norm(x[:, 3:6]),
                        x[:, 6:9] / 1000 + 0.5,
                        # tf.cast(tf.math.rint(x[:, 6:9] / 1000 * RESOLUTION), tf.float32) / RESOLUTION + 0.5,
                        x[:, 9:12],
                        tf.norm(x[:, 3:6], axis=-1, keepdims=True) * tf.random.normal((1,), 1, p_error)
                    ], axis=1), tf.gather_nd(inverse_radiation_length, tf.cast(y[..., tf.newaxis], tf.int32))
                )
            )
            .batch(1)
        )

    ds = construct_ds(10000)

    for x, y in ds: break

    x = x.numpy()
    print(x[:, :, -1])

    output = poca_scattering_density(x[:, :, :3], x[:, :, 3:6], x[:, :, 6:9], x[:, :, 9:12], x[:, :, -1], resolution=16, r=2)
    # print(output)
