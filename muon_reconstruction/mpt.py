import tensorflow as tf
import keras

from keras.models import *
from keras.layers import *


class MostProbableTrajectory(Model):
    def __init__(self, voxels, x, p, ver_x, ver_p):
        self.voxels = voxels

        self.x = x
        self.p = p
        self.ver_x = ver_x
        self.ver_p = ver_p

        self.model = Sequential(
            [
                Dense(128, activation="swish"),
                Dense(64, activation="swish"),
                Dense(2)
            ]
        )

    def call(self, x):
        return self.model(x)

    def train_step(self, data):
        with tf.GradientTape() as t1:
            data = tf.Variable(data)
            with tf.GradientTape() as t2:
                prediction = self(data, training=True)

            dxy_dz = t2.gradient(prediction, data)
            loss = 

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(prediction)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

