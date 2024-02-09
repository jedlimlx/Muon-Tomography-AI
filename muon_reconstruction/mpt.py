import tensorflow as tf

from keras.models import *
from keras.layers import *


class MostProbableTrajectory(Model):
    def __init__(self, voxels, x, p, ver_x, ver_p, num=63, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.voxels = voxels

        self.x = x
        self.p = p
        self.ver_x = ver_x
        self.ver_p = ver_p

        self.z = tf.range(0, num+1, dtype=tf.float32)[::-1][..., None] / num

        self.model = Sequential(
            [
                Dense(64, activation="swish"),
                Dense(128, activation="swish"),
                Dense(2),
            ]
        )

        self.data = None

    def build(self, input_shape):
        self.data = tf.Variable(tf.zeros(input_shape), dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training, mask=mask)

    def train_step(self, data):
        with tf.GradientTape() as t1:
            self.data.assign(data)
            with tf.GradientTape() as t2:
                with tf.GradientTape() as t3:
                    prediction = self(self.data, training=True)

            dxy_dz = t3.batch_jacobian(prediction, self.data)
            dxy_dz = tf.reduce_sum(dxy_dz, axis=-1)

            d2xy_dz2 = t2.batch_jacobian(dxy_dz, self.data)
            d2xy_dz2 = tf.reduce_sum(d2xy_dz2, axis=-1)

            # computing the loss
            numerator = dxy_dz[..., 0] ** 2 + dxy_dz[..., 1] ** 2 + \
                        (dxy_dz[..., 0] * d2xy_dz2[..., 1] - dxy_dz[..., 1] * d2xy_dz2[..., 0]) ** 2
            denominator = (1 + dxy_dz[..., 0] ** 2 + dxy_dz[..., 1] ** 2) ** (5/2)
            loss_1 = tf.reduce_sum(1 / (4 * self.get_density(tf.concat([prediction, self.z], axis=-1))) * numerator / denominator)

            # ensure boundary conditions are adhered to
            y_pred = tf.concat([prediction[0][None, ...], prediction[-1][None, ...]], axis=0)
            y = tf.stop_gradient(tf.concat([self.ver_x[0:2][None, ...], self.x[0:2][None, ...]], axis=0))

            bc_loss = tf.reduce_mean(
                tf.square(dxy_dz[0][0] + self.ver_p[0] / self.ver_p[2])
            ) + tf.reduce_mean(
                tf.square(dxy_dz[0][1] + self.ver_p[1] / self.ver_p[2])
            ) + tf.reduce_mean(
                tf.square(dxy_dz[-1][0] + self.p[0] / self.p[2])
            ) + tf.reduce_mean(
                tf.square(dxy_dz[-1][1] + self.p[1] / self.p[2])
            ) + 0.5 * tf.reduce_mean(tf.square(y - y_pred))

            # compute final loss
            loss = 50 * bc_loss + loss_1

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = t1.gradient(loss, trainable_vars)

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

    def get_density(self, coordinates):
        return 5 * tf.exp(-tf.norm(coordinates - tf.constant([[0.5, 0.5, 0.5]]), axis=1) / 1 ** 2)


if __name__ == "__main__":
    import keras
    import numpy as np
    import matplotlib.pyplot as plt

    from data_generation import predict_trajectory

    batch_size = 64

    data = np.array([[x * 1. / batch_size] for x in range(batch_size)][::-1])
    y_data = np.array([[0.6, 0.8], [0.1, 0.5]])

    # define parameters
    x0 = tf.constant([2567.25, 935.118, 1000]) / 1000
    xf = tf.constant([641.791, 850.859, 1000]) / 1000
    p0 = tf.constant([-1168.68, -51.3007, -1213.95]) / 1685.85862596
    pf = tf.constant([-3, -0.0298983, -0.72007]) / tf.norm(tf.constant([-3, -0.0298983, -0.72007]))

    # get approximate trajectory
    init = tf.concat(
        [
            predict_trajectory(
                x0[0], xf[0], p0[0], pf[0]
            )[..., None],
            predict_trajectory(
                x0[1], xf[1], p0[1], pf[1]
            )[..., None]
        ], axis=-1
    )

    # get most probable trajectory
    model = MostProbableTrajectory(
        0, xf, pf, x0, p0
    )
    model.compile(loss="mse", optimizer=keras.optimizers.AdamW(learning_rate=2e-3, weight_decay=0.0))
    model.build((batch_size, 1))

    model.model.compile(loss="mse", optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.0))
    model.model.fit(data, init, epochs=500, batch_size=batch_size)

    model.fit(data, epochs=4000, batch_size=batch_size, shuffle=False)
