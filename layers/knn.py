from keras.layers import *
from keras.models import *
from keras import ops
import tensorflow as tf


class KNN(Layer):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def call(self, query_points, sample_points, **kwargs):
        query_points = ops.expand_dims(query_points, -2)
        sample_points = ops.expand_dims(sample_points, 1)

        d = ops.sum((query_points - sample_points) * (query_points - sample_points), axis=-1)
        _, indices = ops.top_k(-d, self.k)  # can change to tf.approx_top_k for tpu

        return indices
