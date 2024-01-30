from keras.layers import *
from keras.models import *
from keras import ops
from keras import random
import tensorflow as tf

'''
Adapted from: https://git01lab.cs.univie.ac.at/wolffa95/tpu-k-means.

This file should only be imported after keras backend has been set.
'''


class KMeans(Layer):
    def __init__(self, clusters, dim, iterations, **kwargs):
        super().__init__(**kwargs)

        self.clusters = clusters
        self.dim = dim
        self.iterations = iterations

    def initialize_centroids(self, points):
        # todo replace with keras ops when they come out
        # centroids = random.categorical(ops.ones(ops.shape(points)[:-1]), num_samples=self.clusters)
        # centroids = ops.take_along_axis(points, ops.expand_dims(centroids, -1), axis=1)
        centroids = tf.random.shuffle(tf.transpose(points, [1, 0, 2]))[: self.clusters, ...]
        centroids = tf.transpose(centroids, [1, 0, 2])
        return centroids

    def assign_points(self, points, centroids):
        points = ops.expand_dims(points, axis=2)
        centroids = ops.expand_dims(centroids, axis=1)
        distances = ops.sum((points - centroids) * (points - centroids), axis=-1)
        return ops.argmin(distances, axis=-1)

    def call(self, points, *args, **kwargs):
        # funny stuff to make segment sum work with a batch dimension (a lot of transposing and reshaping)
        b = ops.shape(points)[0]
        offset = ops.expand_dims(ops.arange(b, dtype='int64') * self.clusters, -1)

        centroids = self.initialize_centroids(points)
        points_reshaped = ops.reshape(points, (-1, self.dim))
        for i in range(self.iterations):
            assignments = self.assign_points(points, centroids) + offset
            assignments = ops.reshape(assignments, (-1,))
            centroids = ops.segment_sum(points_reshaped, assignments, num_segments=self.clusters * b, sorted=False)
            centroids /= ops.segment_sum(ops.ones_like(points_reshaped), assignments, num_segments=self.clusters * b,
                                         sorted=False)

            centroids = ops.reshape(centroids, (-1, self.clusters, self.dim))

        return self.assign_points(points, centroids), centroids


def main():
    test_model = Sequential([KMeans(2, 1, 2)])
    test_data = ops.stack([random.normal((10,), mean=-1, stddev=0.5), random.normal((10,), mean=1, stddev=0.5)],
                          axis=-1)
    test_data = ops.reshape(test_data, (1, -1, 1))
    print(test_model(test_data))


if __name__ == "__main__":
    main()
