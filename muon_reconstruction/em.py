import tensorflow as tf
from muon_reconstruction.poca import poca_points, poca


def get_masks(delta_t):
    mask_x = tf.logical_and(delta_t[..., 0] <= delta_t[..., 1], delta_t[..., 0] <= delta_t[..., 2])
    mask_y = tf.logical_and(delta_t[..., 1] < delta_t[..., 0], delta_t[..., 1] <= delta_t[..., 2])
    mask_z = tf.logical_and(delta_t[..., 2] < delta_t[..., 0], delta_t[..., 2] < delta_t[..., 1])
    mask = tf.stack([mask_x, mask_y, mask_z], axis=-1)
    mask_int = tf.cast(mask, tf.int32)
    mask_float = tf.cast(mask, tf.float32)

    return mask_int, mask_float


def voxel_traversal(orig, dirn, t_end, resolution=16):
    visited_voxels = tf.TensorArray(dtype=tf.int32, size=resolution, dynamic_size=True)
    exit_points = tf.TensorArray(dtype=tf.float32, size=resolution, dynamic_size=True)
    path_lengths = tf.TensorArray(dtype=tf.float32, size=resolution, dynamic_size=True)  # L in the paper
    # accumulated_path_lengths = tf.TensorArray(dtype=tf.float32, size=resolution, dynamic_size=True)

    current_position = orig
    current_voxel = tf.cast(tf.floor(orig * resolution), tf.int32)
    t = tf.zeros(orig.shape[:-1])

    unit_step = tf.sign(dirn)
    unit_step_int = tf.cast(unit_step, tf.int32)
    unit_step_relu = tf.nn.relu(unit_step)
    end = orig + t_end * dirn

    cont = t < t_end
    counter = 0
    while tf.reduce_any(cont):
        next_voxel_boundary = (tf.cast(current_voxel, tf.float32) + unit_step_relu) / resolution
        delta_t = (next_voxel_boundary - current_position) / dirn

        mask_int, mask_float = get_masks(delta_t)
        cont = tf.logical_and(t < t_end, cont)
        t = tf.minimum(t_end, t + tf.reduce_min(delta_t, axis=-1))

        exit_point = orig + dirn * t[..., tf.newaxis]
        # exit_point = tf.where((t < t_end)[..., tf.newaxis], orig + dirn * t[..., tf.newaxis], end)
        path_length = tf.math.reduce_euclidean_norm(exit_point - current_position, axis=-1)
        current_position = exit_point

        current_voxel = tf.where(cont[..., tf.newaxis], current_voxel, -1)

        visited_voxels = visited_voxels.write(counter, current_voxel)
        exit_points = exit_points.write(counter, exit_point)
        path_lengths = path_lengths.write(counter, path_length)

        current_voxel += mask_int * unit_step_int

        counter += 1

    rank = len(orig.shape)
    transpose_order = [*list(range(1, rank)), 0, rank]
    visited_voxels = tf.transpose(visited_voxels.stack(), transpose_order)
    exit_points = tf.transpose(exit_points.stack(), transpose_order)
    path_lengths = tf.transpose(path_lengths.stack(), transpose_order[:-1])

    return visited_voxels, exit_points, path_lengths


def expectation_maximization(x, p, ver_x, ver_p, p_est, p_0=13.6, angle_error=0., position_error=0., resolution=64, its=5, **poca_params):
    z_end = tf.math.reduce_mean(x[..., -1])
    z_start = tf.math.reduce_mean(x[..., -1])

    t_int = (z_end - z_start) / ver_p[..., -1]

    x0 = z_start + t_int[..., tf.newaxis] * (z_end - z_start)
    dx = tf.math.reduce_euclidean_norm(x - x0, axis=-1)

    dtheta = tf.math.acos(tf.clip_by_value(tf.reduce_sum(p * ver_p, axis=-1), -1., 1.))

    data = tf.stack([dx, dtheta], axis=-1)[..., tf.newaxis, :, tf.newaxis]  # D vector

    poca_point, _ = poca_points(x, p, ver_x, ver_p)
    dirn_1 = poca_point - ver_x
    dirn_1 = dirn_1 / tf.math.reduce_euclidean_norm(dirn_1, axis=-1, keepdims=True)

    indices_1, exit_points_1, path_lengths_1 = voxel_traversal(x, dirn_1, 1., resolution=resolution)

    dirn_2 = x - poca_point
    dirn_2 = dirn_2 / tf.math.reduce_euclidean_norm(dirn_2, axis=-1, keepdims=True)
    indices_2, exit_points_2, path_lengths_2 = voxel_traversal(poca_point, dirn_2, 1., resolution=resolution)

    poca_voxel = tf.cast(tf.floor(poca_point * resolution), tf.int32)
    poca_mask = tf.cast(tf.reduce_all(indices_1 == poca_voxel[..., tf.newaxis, :], axis=-1), tf.float32)
    poca_l2 = path_lengths_1[..., :1]
    path_lengths_1 += poca_mask * poca_l2

    exit_points_1 += (exit_points_2[..., :1, :] - poca_point[..., tf.newaxis, :]) * poca_mask[..., tf.newaxis]
    remaining_path_lengths_1 = (tf.math.reduce_euclidean_norm(poca_point[..., tf.newaxis, :] - exit_points_1, axis=-1) +
                                tf.math.reduce_euclidean_norm(x - poca_point, axis=-1)[..., tf.newaxis])
    remaining_path_lengths_2 = tf.math.reduce_euclidean_norm(x[..., tf.newaxis, :] - exit_points_2, axis=-1)

    voxel_indices = tf.concat([indices_1, indices_2[..., 1:, :]], axis=-2)
    path_lengths = tf.concat([path_lengths_1, path_lengths_2[..., 1:]], axis=-1)  # L
    # exit_points = tf.concat([exit_points_1, exit_points_2[..., 1:, :]], axis=-2)
    remaining_path_lengths = tf.concat([remaining_path_lengths_1, remaining_path_lengths_2[..., 1:]], axis=-1)  # T

    weights = tf.stack([
        tf.stack([
            path_lengths,
            path_lengths ** 2 / 2 + path_lengths * remaining_path_lengths
        ], axis=-1),
        tf.stack([
            path_lengths ** 2 / 2 + path_lengths * remaining_path_lengths,
            path_lengths ** 3 / 3 + path_lengths ** 2 * remaining_path_lengths + path_lengths * remaining_path_lengths ** 2
        ], axis=-1)
    ], axis=-2)  # W matrix

    # stuff for scatter_nd
    batch_indices = tf.range(x.shape[0], dtype=tf.int32)
    batch_indices = tf.reshape(batch_indices, (-1, 1, 1, 1))
    batch_indices = tf.repeat(batch_indices, axis=1, repeats=voxel_indices.shape[1])
    batch_indices = tf.repeat(batch_indices, axis=2, repeats=voxel_indices.shape[2])
    scatter_indices = tf.concat([batch_indices, voxel_indices], axis=-1)

    scattering_density_voxels = poca(x, p, ver_x, ver_p, p_est, resolution=resolution, **poca_params)
    scattering_density_voxels = tf.transpose(scattering_density_voxels, [0, 2, 1, 3])
#     print("test")
    scattering_density = scattering_density_voxels
    p_rel = (p_est / p_0)[..., tf.newaxis]
    error = tf.convert_to_tensor([
            [angle_error * angle_error, angle_error * position_error],
            [angle_error * position_error, position_error * position_error]
        ], dtype=x.dtype)  # E matrix
    intersect = tf.cast(tf.reduce_all(voxel_indices != -1, axis=-1), tf.float32)
    ray_count = tf.reduce_sum(intersect, axis=-2, keepdims=True)
    for i in range(its):
        print(i)
        scattering_density = tf.gather_nd(params=scattering_density, indices=voxel_indices, batch_dims=1)
        cov = error + p_rel[..., tf.newaxis, tf.newaxis] ** 2 * tf.reduce_sum(scattering_density[..., tf.newaxis, tf.newaxis] * weights, axis=-3, keepdims=True)
        cov_inverse = tf.linalg.pinv(cov)
        conditional_expectation = tf.linalg.matrix_transpose(data) @ cov_inverse @ weights @ cov_inverse @ data
        conditional_expectation = tf.squeeze(conditional_expectation, [-1, -2])
        conditional_expectation = 2 * scattering_density + (conditional_expectation - tf.linalg.trace(cov_inverse @ weights)) * p_rel ** 2 * scattering_density ** 2
        conditional_expectation = conditional_expectation * intersect

        scattering_density = conditional_expectation / (2 * ray_count)
        scattering_density = tf.scatter_nd(indices=scatter_indices, updates=scattering_density, shape=(x.shape[0], resolution, resolution, resolution))

    has_intersection = tf.scatter_nd(indices=scatter_indices, updates=tf.ones_like(path_lengths), shape=(x.shape[0], resolution, resolution, resolution)) != 0.
    return tf.where(has_intersection, scattering_density, scattering_density_voxels)


if __name__ == '__main__':
    from muon_reconstruction.poca import poca_points
    import numpy as np

    # t = np.array([[8.9960396e-01, 3.6855000e-01, -3.9999998e-01, 4.6292034e-01,
    #                8.9384042e-02, -8.8188177e-01, -1.5291035e-02, 1.4414498e-01,
    #                1.4000000e+00, 4.3856499e-01, 1.2083500e-01, -8.9053899e-01,
    #                1.6993467e+03],
    #               [9.8696798e-01, -6.6268027e-02, -3.9999998e-01, -2.4601020e-01,
    #                1.8468954e-01, -9.5150876e-01, 1.4524961e+00, -4.1558695e-01,
    #                1.4000000e+00, -2.4652000e-01, 1.8461600e-01, -9.5139098e-01,
    #                1.1741309e+03],
    #               [4.7593960e-01, 1.0759150e+00, -3.9999998e-01, -1.5553272e-01,
    #                2.3878953e-01, -9.5853502e-01, 7.6863700e-01, 6.2818700e-01,
    #                1.4000000e+00, -1.5627100e-01, 2.3800600e-01, -9.5861000e-01,
    #                9.4515741e+02],
    #               [-4.6373498e-01, -4.3729603e-01, -3.9999998e-01, -6.5669924e-01,
    #                3.3458725e-03, -7.5414515e-01, 1.1042360e+00, -4.4450897e-01,
    #                1.4000000e+00, -6.5704501e-01, 2.8273701e-03, -7.5384599e-01,
    #                1.7999607e+03],
    #               [6.4921498e-01, 4.6886909e-01, -3.9999998e-01, -9.8174796e-02,
    #                4.4913370e-02, -9.9415511e-01, 7.7354199e-01, 4.4384411e-01,
    #                1.4000000e+00, -2.1419700e-02, -3.0903799e-02, -9.9929303e-01,
    #                7.5942297e+02]
    #               ])
    # end = tf.convert_to_tensor(t[..., :3], tf.float32)
    # end_p = tf.convert_to_tensor(t[..., 3:6], tf.float32)
    # orig = tf.convert_to_tensor(t[..., 6:9], tf.float32)
    # orig_p = tf.convert_to_tensor(t[..., 9:12], tf.float32)
    # poca_point, _ = poca_points(end, end_p, orig, orig_p)
    # # print(_)
    # dir1 = poca_point - orig
    # dir2 = end - poca_point
    # orig = tf.concat([orig, poca_point], axis=0)
    # end = tf.concat([poca_point, end], axis=0)
    # dirn = tf.concat([dir1, dir2], axis=0)
    # dirn = dirn / tf.math.reduce_euclidean_norm(dirn, axis=-1, keepdims=True)  # unit vector to make my life easier
    end = tf.convert_to_tensor([[0., 0.2, 0.1]])
    orig = tf.convert_to_tensor([[0.001, 0.9, 0.8]])
    dirn = end - orig
    t_max = 1.
    print(orig, dirn, t_max)
    # print(orig)
    # print(end)
    # print(poca_point)
    # print(dirn)

    voxel_indices, exit_points, path_lengths = voxel_traversal(orig, dirn, t_max, 16)
    # voxel_indices = tf.reshape(voxel_indices, (-1, 3))
    voxels = tf.scatter_nd(voxel_indices, tf.ones(voxel_indices.shape[:-1]), shape=(16, 16, 16))

    # print(voxels.shape)

    import matplotlib.pyplot as plt

    plt.imshow(voxels[0].numpy(), origin='lower')
    plt.plot([0.8 * 16 - 0.5, 0.1 * 16 - 0.5], [0.9 * 16 - 0.5, 0.2 * 16 - 0.5])
    plt.show()

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels.numpy())
    plt.show()
    # plt.plot(voxel_indices[0, :, 0])
    # plt.show()
    # plt.plot(voxel_indices[0, :, 1])
    # plt.show()
    # plt.plot(voxel_indices[0, :, 2])
    # plt.show()
    # print(orig * 64)
    # print(end * 64)
    # print(tf.reduce_max(tf.clip_by_value(voxel_indices[1, :, 2], 0, 100)))
    # print(tf.reduce_min(tf.where(voxel_indices[0, :, 2] == -1, 100, voxel_indices[0, :, 2])))

    # plt.plot(voxel_indices[1, :, 0])
    # plt.show()
    # plt.plot(voxel_indices[2, :, 0])
    # plt.show()
    # plt.plot(voxel_indices[3, :, 0])
    # plt.show()
