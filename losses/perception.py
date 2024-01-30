import tensorflow as tf
from keras.losses import MeanSquaredError
from keras.applications.efficientnet import EfficientNetB3

model = EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_shape=None
)


def perception_loss(y_pred, y_true, model, max_value=1.6):
    shape = tf.shape(y_pred)
    y_pred = tf.math.divide(tf.reshape(y_pred, (shape[0] * shape[1], shape[2], shape[3], shape[4])), max_value / 255)
    y_true = tf.math.divide(tf.reshape(y_true, (shape[0] * shape[1], shape[2], shape[3], shape[4])), max_value / 255)

    feature_pred = model(tf.tile(y_pred, tf.constant([1, 1, 1, 3])))
    feature_true = model(tf.tile(y_true, tf.constant([1, 1, 1, 3])))

    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    return mse(feature_pred, feature_true)


if __name__ == "__main__":
    import numpy as np
    print(perception_loss(np.zeros((1, 2, 256, 256, 1)), np.ones((1, 2, 256, 256, 1))))
