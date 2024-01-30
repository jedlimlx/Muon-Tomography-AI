from keras.layers import *
from layers import KMeans
from layers import KNN
from keras import ops


class KNNDownSampling(Layer):
    def __init__(self, centroids, points_per_centroid, sampling_mode='random', pooling_mode='avg',
                 kmeans_its=5, **kwargs):
        super().__init__(**kwargs)
        self.k_means = None
        self.centroids = centroids
        self.points_per_centroid = points_per_centroid
        self.sampling_mode = sampling_mode
        self.pooling_mode = pooling_mode
        self.kmeans_its = kmeans_its

        self.knn = KNN(points_per_centroid)

        if pooling_mode == 'avg':
            self.pooling = AveragePooling1D(pool_size=points_per_centroid)
        elif pooling_mode == 'max':
            self.pooling = MaxPooling1D(pool_size=points_per_centroid)

    def build(self, input_shape):
        # print(input_shape)
        _, s, d = input_shape[1]
        self.k_means = KMeans(self.centroids, d, self.kmeans_its)

    def call(self, inputs, *args, **kwargs):
        x, positions = inputs
        _, c = self.k_means(positions)
        # print(self.centroids)
        # print(c.shape, positions.shape)
        cluster_idx = self.knn(c, positions)
        cluster_idx = ops.expand_dims(cluster_idx, -1)
        cluster_idx = ops.reshape(cluster_idx, (-1, self.centroids * self.points_per_centroid, 1))
        # print(cluster_idx.shape, x.shape)
        x = ops.take_along_axis(x, cluster_idx, 1)
        return self.pooling(x)


class WindowAttention(Layer):
    """
    From: https://keras.io/examples/vision/swin_transformers/#window-based-multihead-selfattention
    """
    def __init__(self, dim, window_size, heads, qkv_bias=True, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_dense = Dense(dim, use_bias=qkv_bias, name=f'{self.name}-q_dense')
        self.k_dense = Dense(dim, use_bias=qkv_bias, name=f'{self.name}-k_dense')
        self.v_dense = Dense(dim, use_bias=qkv_bias, name=f'{self.name}-v_dense')
        self.dropout_1 = Dropout(dropout, name=f'{self.name}-attn_drop')
        self.dropout_2 = Dropout(dropout, name=f'{self.name}-proj_drop')
        self.proj = Dense(dim, name=f'{self.name}-proj_dense')

    def call(self, x, *args, **kwargs):
        _, windows, window_size, d = x.shape
        head_dim = d // self.heads
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)

        q = ops.reshape(q, (-1, windows, self.window_size, self.heads, head_dim))
        k = ops.reshape(k, (-1, windows, self.window_size, self.heads, head_dim))
        v = ops.reshape(v, (-1, windows, self.window_size, self.heads, head_dim))

        q = ops.transpose(q, (0, 1, 3, 2, 4))
        k = ops.transpose(k, (0, 1, 3, 4, 2))
        v = ops.transpose(v, (0, 1, 3, 2, 4))

        attn = ops.softmax(q @ k)
        attn = self.dropout_1(attn)

        x_attn = attn @ v
        x_attn = ops.transpose(x_attn, (0, 1, 3, 2, 4))
        x_attn = ops.reshape(x_attn, (-1, windows, window_size, d))
        x_attn = self.proj(x_attn)
        x_attn = self.dropout_2(x_attn)
        return x_attn


def main():
    # features = np.random.uniform(size=(16384, 2))
    # downsampling = KNNDownSampling(int(16384 / 256), 256)
    # print(downsampling([features[tf.newaxis, ...], features[tf.newaxis, ...]]).shape)
    test_wmhsa = WindowAttention(512, 256, 16)
    print(test_wmhsa(ops.ones((1, 64, 256, 512))))


if __name__ == '__main__':
    main()
