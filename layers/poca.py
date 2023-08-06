import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from functools import partial


def poca(x, p, ver_x, ver_p, nn=None):
    if nn is None:
        v = tf.linalg.cross(p, ver_p)

        m = tf.stack([p, -ver_p, v], axis=-1)
        b = ver_x - x

        m_inv = tf.linalg.pinv(m)
        t = tf.linalg.matmul(m_inv, b[..., tf.newaxis])
        scattered = (tf.linalg.det(m) > 2e-6) | (tf.linalg.det(m) < -2e-6)
        not_scattered = tf.cast(~scattered, tf.float32)[..., tf.newaxis]
        scattered = tf.cast(scattered, tf.float32)[..., tf.newaxis]
        t, ver_t, _ = tf.unstack(tf.squeeze(t, axis=-1), axis=-1)
        t = t[..., tf.newaxis]
        ver_t = ver_t[..., tf.newaxis]

        poca_points = (x + p * t + ver_x + ver_t * ver_p) / 2
        poca_points = poca_points * scattered  # + (x + ver_x) / 2 * not_scattered

        return poca_points
    else:
        output = nn(tf.concat([x, p, ver_x, ver_p], axis=-1), training=False)

        n = tf.math.rint(output[:, :, 0:1])
        mask = (output * 0 + tf.range(16, dtype=tf.float32)) <= 3 * tf.repeat(n, 16, axis=-1)

        masked_output = output * tf.cast(mask, tf.float32)
        return tf.concat([
            masked_output[:, :, 1:4],
            masked_output[:, :, 4:7],
            masked_output[:, :, 7:10],
            masked_output[:, :, 10:13],
            masked_output[:, :, 13:16]
        ], axis=1)


class EncoderBlock(Layer):
    def __init__(self, num_heads=16, dim=256, mlp_units=512, dropout=0., out_dim=None, activation='gelu',
                 name='vit_block', norm=partial(LayerNormalization, epsilon=1e-5), **kwargs):
        super().__init__(name=name, **kwargs)
        if out_dim is None:
            out_dim = dim

        self.num_heads = num_heads
        self.dim = dim
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.out_dim = out_dim
        self.activation = activation

        self.norm1 = norm(name=f"{name}_norm_1")
        self.mhsa = MultiHeadAttention(num_heads=num_heads, key_dim=dim, dropout=dropout, name=f"{name}_mha")
        self.norm2 = norm(name=f"{name}_norm_2")
        self.mlp = Sequential([
            Dense(mlp_units, activation=activation),
            Dense(out_dim)
        ], name=f"{name}_mlp")

    def call(self, inputs, **kwargs):
        x1 = self.norm1(inputs)
        attn = self.mhsa(x1, x1)
        x2 = attn + x1
        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        return x2 + x3

    @classmethod
    def from_config(cls, config):
        norm_cls = deserialize(config['norm']).__class__
        del config['norm']['config']['name']
        norm = partial(norm_cls, **config['norm']['config'])
        del config['norm']

        return cls(**config, norm=norm)

    def get_config(self):
        cfg = super(EncoderBlock, self).get_config()
        cfg.update({
            'num_heads': self.num_heads,
            'dim': self.dim,
            'mlp_units': self.mlp_units,
            'dropout': self.dropout,
            'out_dim': self.out_dim,
            'activation': self.activation,
            'norm': serialize(self.norm1)
        })

        return cfg


class PoCATransformer(Model):
    def __init__(self, num_layers=3, dim=64, mlp_units=64, k=5, num_heads=16, *args, **kwargs):
        super(PoCATransformer, self).__init__(*args, **kwargs)

        self.num_layers = num_layers
        self.dim = dim
        self.mlp_units = mlp_units
        self.k = k

        self.input_projection = Dense(self.dim, activation="gelu")
        self.output_projection = Dense(1+3*self.k)

        # intermediate layers
        self.enc_layers = [
            EncoderBlock(num_heads=num_heads, dim=dim, mlp_units=mlp_units) for _ in range(num_layers)
        ]

    def call(self, muons, training=None, mask=None):
        x = self.input_projection(muons)

        for layer in self.enc_layers:
            x = layer(x)

        return self.output_projection(x)


# a neural network version of poca :)
class PoCAModel(Model):
    def __init__(self, num_layers=3, d=64, k=5, *args, **kwargs):
        super(PoCAModel, self).__init__(*args, **kwargs)

        self.num_layers = num_layers
        self.d = d
        self.k = k

        # intermediate layers
        self.nn = [
            Sequential([
                Dense(4 * self.d, activation="gelu", name=f"{self.name}_hidden_layer_{i}"),
                Dense(self.d, name=f"{self.name}_projection_{i}")
            ]) for i in range(self.num_layers)
        ]

        self.final_layer = Dense(1 + 3 * self.k)

    def call(self, muons, training=None, mask=None):
        lst = []

        x = muons
        for i in range(self.num_layers):
            x = self.nn[i](x)

            lst.append(x)

            feature_vector = tf.math.reduce_mean(x, axis=1, keepdims=True)
            x = tf.concat(lst + [x * 0 + feature_vector], axis=-1)  # do some h3x to handle the ragged batch

        return self.final_layer(x)


def loss(y, y_pred):
    n = y[:, :, 0:1]
    # more h3x to handle the ragged batch
    mask = (y * 0 + tf.range(16, dtype=tf.float32)) <= 3 * tf.repeat(n, 16, axis=-1)
    return tf.math.reduce_sum(tf.square(y - y_pred) * tf.cast(mask, tf.float32)) / \
        tf.math.reduce_sum(tf.cast(mask, tf.float32))


if __name__ == "__main__":
    # Create a description of the features.
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
        'voxels': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_example(example_proto):
        res = tf.io.parse_single_example(example_proto, feature_description)
        x = tf.io.parse_tensor(res['x'], out_type=tf.double)
        y = tf.io.parse_tensor(res['y'], out_type=tf.double)
        voxels = tf.io.parse_tensor(res['voxels'], out_type=tf.int32)

        x.set_shape((None, 12))
        y.set_shape((None, 1+5*3))
        voxels.set_shape((64, 64, 64))

        return tf.cast(x[:750], tf.float32), tf.cast(y[:750], tf.float32)

    ds = tf.data.TFRecordDataset("../scattering_prediction.tfrecord").map(_parse_example).batch(16)
    val_ds = ds.take(10)
    train_ds = ds.skip(10)

    """
    # building model
    model = PoCATransformer(num_layers=1, dim=64, num_heads=4)
    model.compile(optimizer="adam", loss=loss)

    model.fit(train_ds, epochs=30, validation_data=val_ds)
    """

    for x, y in val_ds.take(1): break

    output = y
    print(tf.gather_nd(output, tf.where(output[:, :, 0] > 0.5)).shape)
    print(
        tf.concat([
            tf.gather_nd(output, tf.where(output[:, :, 0] > 0.5))[:, 1:4],
            tf.gather_nd(output, tf.where(output[:, :, 0] > 1.5))[:, 4:7],
            tf.gather_nd(output, tf.where(output[:, :, 0] > 2.5))[:, 7:10],
            tf.gather_nd(output, tf.where(output[:, :, 0] > 3.5))[:, 10:13],
            tf.gather_nd(output, tf.where(output[:, :, 0] > 4.5))[:, 13:16]
        ], axis=0)
    )
