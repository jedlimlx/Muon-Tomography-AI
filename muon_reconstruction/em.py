import tensorflow as tf


def voxel_traversal(orig, dirn, end, resolution=64):
    visited_voxels = tf.TensorArray(dtype=tf.int32, size=resolution, dynamic_size=True)
    current_voxel = tf.cast(orig * resolution, tf.int32)
    end_voxel = tf.cast(end * resolution, tf.int32)
    end_t = (end - orig) / dirn
    step = tf.sign(dirn)

    next_voxel_boundary = (tf.cast(current_voxel, tf.float32) + step) / resolution

    t_max = (next_voxel_boundary - orig) / dirn
    t_delta = step / resolution / dirn

    step = tf.cast(step, tf.int32)

    cont = tf.math.logical_not(tf.math.reduce_all(current_voxel == end_voxel, axis=-1))

    counter = 0
    visited_voxels = visited_voxels.write(counter, current_voxel)
    # print(max_steps)

    while tf.reduce_any(cont):
        mask_x = tf.logical_and(t_max[..., 0] <= t_max[..., 1], t_max[..., 0] <= t_max[..., 2])
        mask_y = tf.logical_and(t_max[..., 1] < t_max[..., 0], t_max[..., 1] <= t_max[..., 2])
        mask_z = tf.logical_and(t_max[..., 2] < t_max[..., 0], t_max[..., 2] < t_max[..., 1])

        # print(t_max)
        mask_x = tf.logical_and(mask_x, cont)
        mask_y = tf.logical_and(mask_y, cont)
        mask_z = tf.logical_and(mask_z, cont)

        mask = tf.stack([mask_x, mask_y, mask_z], axis=-1)
        mask_float = tf.cast(mask, orig.dtype)
        mask_int = tf.cast(mask, tf.int32)

        current_voxel += mask_int * step
        t_max += mask_float * t_delta

        # print(t_max)

        counter += 1
        visited_voxels = visited_voxels.write(counter, current_voxel)

        # print(current_voxel, end_voxel
        # print((end_voxel - current_voxel) * step)
        cont = tf.math.logical_not(tf.math.reduce_all((end_voxel - current_voxel) * step < 0, axis=-1))

        # print(current_voxel)
        # print(end_voxel)
        # print(cont)
        # print(counter)

    return tf.transpose(visited_voxels.stack(), [1, 0, 2])


def expectation_maximization(x, ver_x, p, ver_p, p_est, resolution=64, its=5):
    b, s, _ = x.shape


if __name__ == '__main__':
    from muon_reconstruction.poca import poca_points
    import numpy as np

    t = np.array([[8.9960396e-01, 3.6855000e-01, -3.9999998e-01, 4.6292034e-01,
                   8.9384042e-02, -8.8188177e-01, -1.5291035e-02, 1.4414498e-01,
                   1.4000000e+00, 4.3856499e-01, 1.2083500e-01, -8.9053899e-01,
                   1.6993467e+03],
                  [9.8696798e-01, -6.6268027e-02, -3.9999998e-01, -2.4601020e-01,
                   1.8468954e-01, -9.5150876e-01, 1.4524961e+00, -4.1558695e-01,
                   1.8468954e-01, -9.5150876e-01, 1.4524961e+00, -4.1558695e-01,
                   1.4000000e+00, -2.4652000e-01, 1.8461600e-01, -9.5139098e-01,
                   1.1741309e+03],
                  [4.7593960e-01, 1.0759150e+00, -3.9999998e-01, -1.5553272e-01,
                   2.3878953e-01, -9.5853502e-01, 7.6863700e-01, 6.2818700e-01,
                   1.4000000e+00, -1.5627100e-01, 2.3800600e-01, -9.5861000e-01,
                   9.4515741e+02],
                  [-4.6373498e-01, -4.3729603e-01, -3.9999998e-01, -6.5669924e-01,
                   3.3458725e-03, -7.5414515e-01, 1.1042360e+00, -4.4450897e-01,
                   1.4000000e+00, -6.5704501e-01, 2.8273701e-03, -7.5384599e-01,
                   1.7999607e+03],
                  [6.4921498e-01, 4.6886909e-01, -3.9999998e-01, -9.8174796e-02,
                   4.4913370e-02, -9.9415511e-01, 7.7354199e-01, 4.4384411e-01,
                   1.4000000e+00, -2.1419700e-02, -3.0903799e-02, -9.9929303e-01,
                   7.5942297e+02]
                  ])
    end = tf.convert_to_tensor(t[..., :3], tf.float32)
    end_p = tf.convert_to_tensor(t[..., 3:6], tf.float32)
    orig = tf.convert_to_tensor(t[..., 6:9], tf.float32)
    orig_p = tf.convert_to_tensor(t[..., 9:12], tf.float32)
    poca_point, _ = poca_points(end, end_p, orig, orig_p)
    # print(_)
    dir1 = poca_point - orig
    dir2 = end - poca_point
    orig = tf.concat([orig, poca_point], axis=0)
    end = tf.concat([poca_point, end], axis=0)
    dirn = tf.concat([dir1, dir2], axis=0)
    # print(orig)
    # print(end)
    # print(poca_point)
    # print(dirn)

    voxel_indices = voxel_traversal(orig, dirn, end, 64)
    # voxel_indices = tf.reshape(voxel_indices, (-1, 3))
    voxels = tf.scatter_nd(voxel_indices, tf.ones(voxel_indices.shape[:-1]), shape=(64, 64, 64))
    # print(voxels.shape)

    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels.numpy())
    plt.show()
