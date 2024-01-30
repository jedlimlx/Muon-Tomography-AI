import tensorflow as tf
from muon_reconstruction.poca import poca_points, poca


def get_masks(delta_t):
    mask_x = tf.logical_and(delta_t[..., 0] <= delta_t[..., 1], delta_t[..., 0] <= delta_t[..., 2])
    mask_y = tf.logical_and(delta_t[..., 1] < delta_t[..., 0], delta_t[..., 1] <= delta_t[..., 2])
    mask_z = tf.logical_and(delta_t[..., 2] < delta_t[..., 0], delta_t[..., 2] < delta_t[..., 1])
    mask = tf.stack([mask_x, mask_y, mask_z], axis=-1)
    mask_int = tf.cast(mask, tf.int32)
    mask_float = tf.cast(mask, delta_t.dtype)

    return mask_int, mask_float


def voxel_traversal(orig, dirn, t1, resolution=16):
    visited_voxels = tf.TensorArray(dtype=tf.int32, size=resolution, dynamic_size=True)
    path_lengths = tf.TensorArray(dtype=orig.dtype, size=resolution, dynamic_size=True)  # L in the paper
    cont_arr = tf.TensorArray(dtype=orig.dtype, size=resolution, dynamic_size=True)
    # accumulated_path_lengths = tf.TensorArray(dtype=tf.float32, size=resolution, dynamic_size=True)

    t_start = (tf.cast(dirn < 0, orig.dtype) - orig) / dirn
    t_start = tf.where(tf.math.is_finite(t_start), t_start, t_start.dtype.min)
    t_start = tf.reduce_max(t_start, axis=-1)
    t_start = tf.maximum(t_start, 0)
    t_end = (tf.cast(dirn > 0, orig.dtype) - orig) / dirn
    t_end = tf.where(tf.math.is_finite(t_end), t_end, t_end.dtype.max)
    t_end = tf.reduce_min(t_end, axis=-1)
    t_end = tf.minimum(t1, t_end)
    current_position = orig + dirn * t_start[..., tf.newaxis]
    current_voxel = tf.cast(tf.floor(current_position * resolution), tf.int32)
    t = t_start

    unit_step = tf.sign(dirn)
    unit_step_int = tf.cast(unit_step, tf.int32)
    unit_step_relu = tf.nn.relu(unit_step)

    cont = t < t_end
    counter = 0
    while tf.reduce_any(cont) or counter < resolution - 1:
        next_voxel_boundary = (tf.cast(current_voxel, orig.dtype) + unit_step_relu) / resolution
        delta_t = (next_voxel_boundary - current_position) / dirn
        delta_t = tf.where(tf.math.is_finite(delta_t), delta_t, delta_t.dtype.max)

        mask_int, mask_float = get_masks(delta_t)
        cont = tf.logical_and(t < t_end, cont)
        cont_float = tf.cast(cont, orig.dtype)
        t = tf.minimum(t_end, t + tf.reduce_min(delta_t, axis=-1))
        exit_point = orig + dirn * t[..., tf.newaxis]
        # exit_point = tf.where((t < t_end)[..., tf.newaxis], orig + dirn * t[..., tf.newaxis], end)
        path_length = tf.math.reduce_euclidean_norm(exit_point - current_position, axis=-1)
        current_position = exit_point

        current_voxel = tf.where(cont[..., tf.newaxis], current_voxel, -1)
        path_length = cont_float * path_length

        visited_voxels = visited_voxels.write(counter, current_voxel)
        path_lengths = path_lengths.write(counter, path_length)
        cont_arr = cont_arr.write(counter, cont_float)

        current_voxel += mask_int * unit_step_int

        counter += 1

    rank = len(orig.shape)
    transpose_order = [*list(range(1, rank)), 0, rank]
    visited_voxels = tf.transpose(visited_voxels.stack(), transpose_order)
    path_lengths = tf.transpose(path_lengths.stack(), transpose_order[:-1])
    cont_arr = tf.transpose(cont_arr.stack(), transpose_order[:-1])

    return visited_voxels, path_lengths, cont_arr


def get_ray_data(inp):
    x_out, y_out, z_out, px_out, py_out, pz_out, x_in, y_in, z_in, px_in, py_in, pz_in, p_est = tf.unstack(
        inp[..., :-1], axis=-1)

    theta_x_in = tf.atan(px_in / pz_in)
    theta_y_in = tf.atan(py_in / pz_in)
    theta_x_out = tf.atan(px_out / pz_out)
    theta_y_out = tf.atan(py_out / pz_out)

    # x and y scattering angle
    d_theta_x = theta_x_out - theta_x_in
    d_theta_y = theta_y_out - theta_y_in

    # average x and y ray angle
    avg_theta_x = (theta_x_in + theta_x_out) / 2
    avg_theta_y = (theta_y_in + theta_y_out) / 2

    # projected x and y position
    x_proj = x_in + tf.tan(theta_x_in) * (z_out - z_in)
    y_proj = y_in + tf.tan(theta_y_in) * (z_out - z_in)

    # adjustment factors for 3D path length
    f_xy = tf.tan(theta_x_in) ** 2 + tf.tan(theta_y_in) ** 2 + 1
    f_x = tf.tan(theta_x_in) ** 2 + 1
    f_y = tf.tan(theta_y_in) ** 2 + 1

    # x and y displacement
    # the original author gives 2 different expressions for this in his 2007 paper and PhD thesis
    # this is the expression from the PhD dissertation. they are very similar in value
    d_x = (x_out - x_proj) * tf.sqrt(f_xy / f_x) * tf.cos(avg_theta_x)
    d_y = (y_out - y_proj) * tf.sqrt(f_xy / f_y) * tf.cos(avg_theta_y)
    p_rel = (15 / p_est) ** 2
    return tf.stack([d_theta_x, d_theta_y, d_x, d_y, p_rel], axis=-1)


# convenience class. there's no error handling at all and only multiply, inverse and trace is implemented
class Matrix:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]))

    def __matmul__(self, other):
        result = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(other.data[0])):
                acc = 0
                for k in range(len(other.data)):
                    acc = acc + self.data[i][k] * other.data[k][j]
                row.append(acc)
            result.append(row)
        return Matrix(result)

    def __mul__(self, other):
        return Matrix([[self.data[i][j] * other for j in range(len(self.data[i]))] for i in range(self.data)])

    def __rmul__(self, other):
        return Matrix([[self.data[i][j] * other for j in range(len(self.data[i]))] for i in range(len(self.data))])

    @property
    def inverse(self):
        det = self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        return Matrix([[self.data[1][1] / det, -self.data[0][1] / det],
                       [-self.data[1][0] / det, self.data[0][0] / det]])

    @property
    def trace(self):
        return Matrix([[self.data[0][0] + self.data[1][1]]])


# %%
class SparseHelper:
    def __init__(self, tensor):
        self.tensor = tensor

    def __mul__(self, other):
        try:
            return SparseHelper(self.tensor * other)
        except ValueError:
            return SparseHelper(tf.sparse.map_values(tf.multiply, self.tensor, other))

    def __rmul__(self, other):
        try:
            return SparseHelper(self.tensor * other)
        except ValueError:
            return SparseHelper(tf.sparse.map_values(tf.multiply, self.tensor, other))

    def __pow__(self, power, modulo=None):
        return SparseHelper(tf.sparse.map_values(tf.pow, self.tensor, power))

    def __add__(self, other):
        if isinstance(other, tf.Tensor):
            return SparseHelper(tf.sparse.add(self.tensor, other))
        elif isinstance(other, SparseHelper):
            return SparseHelper(tf.sparse.map_values(tf.add, self.tensor, other.tensor))
        else:
            return SparseHelper(tf.sparse.map_values(tf.add, self.tensor, other))

    def __radd__(self, other):
        if isinstance(other, tf.Tensor):
            return SparseHelper(tf.sparse.add(self.tensor, other))
        elif isinstance(other, SparseHelper):
            return SparseHelper(tf.sparse.map_values(tf.add, self.tensor, other.tensor))
        else:
            return SparseHelper(tf.sparse.map_values(tf.add, self.tensor, other))

    def __sub__(self, other):
        if isinstance(other, tf.Tensor):
            return SparseHelper(tf.sparse.add(self.tensor, -other))
        elif isinstance(other, SparseHelper):
            return SparseHelper(tf.sparse.map_values(tf.math.subtract, self.tensor, other.tensor))
        else:
            return SparseHelper(tf.sparse.map_values(tf.math.subtract, self.tensor, other))

    def __truediv__(self, other):
        try:
            return SparseHelper(self.tensor / other)
        except ValueError:
            return SparseHelper(tf.sparse.map_values(tf.divide, self.tensor, other))


# %%
def mlem(inp, resolution=64, det_resolution=None, lambda_init=None, its=5):
    if lambda_init is None:
        lambda_init = tf.fill((tf.shape(inp)[0], 1, resolution, resolution, resolution),
                              tf.constant(1 / 3039, dtype=inp.dtype))
    scattering_density = lambda_init
    data = get_ray_data(inp)
    p_r = data[..., -1, tf.newaxis, tf.newaxis, tf.newaxis]

    # data vectors
    data_x = Matrix([
        [data[..., 0][..., tf.newaxis, tf.newaxis, tf.newaxis]],
        [data[..., 2][..., tf.newaxis, tf.newaxis, tf.newaxis]]
    ])
    data_x_t = Matrix([[
        data[..., 0][..., tf.newaxis, tf.newaxis, tf.newaxis],
        data[..., 2][..., tf.newaxis, tf.newaxis, tf.newaxis]
    ]])
    data_y = Matrix([
        [data[..., 1][..., tf.newaxis, tf.newaxis, tf.newaxis]],
        [data[..., 3][..., tf.newaxis, tf.newaxis, tf.newaxis]]
    ])
    data_y_t = Matrix([[
        data[..., 1][..., tf.newaxis, tf.newaxis, tf.newaxis],
        data[..., 3][..., tf.newaxis, tf.newaxis, tf.newaxis]
    ]])

    end = inp[..., :3]
    start = inp[..., 6:9]
    poca = poca_points(inp[..., :3], inp[..., 3:6], inp[..., 6:9], inp[..., 9:12])[0]

    dirn_1 = poca - start

    indices_1, path_lengths_1, cont_1 = voxel_traversal(start, dirn_1, 1., resolution=resolution)

    dirn_2 = end - poca
    indices_2, path_lengths_2, cont_2 = voxel_traversal(poca, dirn_2, 1., resolution=resolution)

    poca_voxel = tf.cast(tf.floor(poca * resolution), tf.int32)
    poca_mask = tf.cast(tf.reduce_all(indices_1 == poca_voxel[..., tf.newaxis, :], axis=-1), tf.float32)
    poca_l2 = path_lengths_2[..., :1]
    path_lengths_1 += poca_mask * poca_l2

    voxel_indices = tf.concat([indices_1, indices_2[..., 1:, :]], axis=-2)
    path_lengths = tf.concat([path_lengths_1, path_lengths_2[..., 1:]], axis=-1)  # L
    remaining_path_lengths = tf.cumsum(path_lengths, axis=1, exclusive=True, reverse=True)  # T

    weight_11 = path_lengths
    weight_12 = (path_lengths ** 2 / 2 + path_lengths * remaining_path_lengths) * tf.sign(path_lengths)
    weight_21 = weight_12
    weight_22 = (
                            path_lengths ** 3 / 2 + path_lengths ** 2 * remaining_path_lengths + path_lengths * remaining_path_lengths ** 2) * tf.sign(
        path_lengths)

    batch_idx = tf.range(tf.shape(voxel_indices)[0])
    batch_idx = batch_idx[:, tf.newaxis, tf.newaxis, tf.newaxis]
    batch_idx = tf.repeat(batch_idx, tf.shape(voxel_indices)[1], axis=1)
    batch_idx = tf.repeat(batch_idx, tf.shape(voxel_indices)[2], axis=2)
    point_idx = tf.range(tf.shape(indices_1)[1])
    point_idx = point_idx[tf.newaxis, :, tf.newaxis, tf.newaxis]
    point_idx = tf.repeat(point_idx, tf.shape(voxel_indices)[0], axis=0)
    point_idx = tf.repeat(point_idx, tf.shape(voxel_indices)[2], axis=2)

    idx = tf.concat([batch_idx, point_idx, voxel_indices], axis=-1)
    idx = tf.cast(tf.reshape(idx, (-1, 5)), tf.int64)
    weight_11 = tf.reshape(weight_11, (-1,))
    weight_12 = tf.reshape(weight_12, (-1,))
    weight_21 = tf.reshape(weight_21, (-1,))
    weight_22 = tf.reshape(weight_22, (-1,))

    mask = tf.reduce_all((idx[:, 2:] >= 0) & (idx[:, 2:] < resolution), axis=-1)
    idx = tf.boolean_mask(idx, mask, axis=0)
    weight_11 = tf.boolean_mask(weight_11, mask, axis=0)
    weight_12 = tf.boolean_mask(weight_12, mask, axis=0)
    weight_21 = tf.boolean_mask(weight_21, mask, axis=0)
    weight_22 = tf.boolean_mask(weight_22, mask, axis=0)

    weight_11 = tf.sparse.SparseTensor(idx, weight_11, dense_shape=(
    tf.shape(inp)[0], tf.shape(inp)[1], resolution, resolution, resolution))
    weight_11 = SparseHelper(tf.sparse.reorder(weight_11))
    weight_12 = tf.sparse.SparseTensor(idx, weight_12, dense_shape=(
    tf.shape(inp)[0], tf.shape(inp)[1], resolution, resolution, resolution))
    weight_12 = SparseHelper(tf.sparse.reorder(weight_12))
    weight_21 = tf.sparse.SparseTensor(idx, weight_21, dense_shape=(
    tf.shape(inp)[0], tf.shape(inp)[1], resolution, resolution, resolution))
    weight_21 = SparseHelper(tf.sparse.reorder(weight_21))
    weight_22 = tf.sparse.SparseTensor(idx, weight_22, dense_shape=(
    tf.shape(inp)[0], tf.shape(inp)[1], resolution, resolution, resolution))
    weight_22 = SparseHelper(tf.sparse.reorder(weight_22))

    ones = weight_11.tensor.with_values(tf.sign(weight_11.tensor.values))

    weights = Matrix([
        [weight_11, weight_12],
        [weight_21, weight_22]
    ])

    error = 1 / det_resolution if det_resolution is not None else 0
    for it in range(its):
        std_dev_11 = p_r * tf.sparse.reduce_sum((weight_11 * scattering_density).tensor, axis=[2, 3, 4], keepdims=True)
        std_dev_12 = p_r * tf.sparse.reduce_sum((weight_12 * scattering_density).tensor, axis=[2, 3, 4], keepdims=True)
        std_dev_21 = p_r * tf.sparse.reduce_sum((weight_21 * scattering_density).tensor, axis=[2, 3, 4], keepdims=True)
        std_dev_22 = (error ** 2) / 12 + p_r * tf.sparse.reduce_sum((weight_22 * scattering_density).tensor,
                                                                    axis=[2, 3, 4], keepdims=True)

        std_dev = Matrix([
            [std_dev_11, std_dev_12],
            [std_dev_21, std_dev_22]
        ])
        print(data_x_t @ std_dev.inverse @ weights)
        std_inv = Matrix([[tf.where(tf.math.is_finite(std_dev.inverse.data[i][j]), std_dev.inverse.data[i][j], 0) for j
                           in range(len(std_dev.inverse.data[i]))] for i in range(len(std_dev.inverse.data))])
        s_x = SparseHelper(2 * scattering_density * ones) + (
                    (data_x_t @ std_inv @ weights @ std_inv @ data_x).data[0][0] -
                    (std_inv @ weights).trace.data[0][0]) * p_r * scattering_density ** 2
        s_y = SparseHelper(2 * scattering_density * ones) + (
                    (data_y_t @ std_inv @ weights @ std_inv @ data_y).data[0][0] -
                    (std_inv @ weights).trace.data[0][0]) * p_r * scattering_density ** 2

        s = (s_x + s_y) / 2
        scattering_density = 0.5 * tf.math.divide_no_nan(tf.sparse.reduce_sum(s.tensor, axis=1, keepdims=True),
                                                         tf.sparse.reduce_sum(ones, axis=1, keepdims=True))
