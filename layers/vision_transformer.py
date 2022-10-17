import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def MLP(hidden_units, dropout_rate, activation='swish', name=None):
    model = Sequential(name=name)

    i = 0
    for units in hidden_units:
        model.add(Dense(units, activation=activation, name=name + f"_dense_{i}"))
        model.add(Dropout(dropout_rate, name=name + f"_dropout_{i}"))
        i += 1

    return model


def ViTBlock(num_heads, projection_dim, transformer_units, name=None):
    def apply(encoded_patches):
        # Layer normalisation 1
        x1 = LayerNormalization(epsilon=1e-6, name=f"{name}_layer_norm_1")(encoded_patches)

        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=0.1,
            name=f"{name}_mhsa"
        )(x1, x1)

        # Skip connection 1
        x2 = Add(name=f"{name}_add_1")([attention_output, encoded_patches])

        # Layer Normalisation 2
        x3 = LayerNormalization(epsilon=1e-6, name=f"{name}_layer_norm_2")(x2)

        # Multi-layer perception
        x3 = MLP(hidden_units=transformer_units, dropout_rate=0.1, name=f"{name}_mlp")(x3)

        # Skip connection 2
        encoded_patches = Add(name=f"{name}_add_2")([x3, x2])
        return encoded_patches

    return apply


class Patches(Layer):
    def __init__(self, patch_width, patch_height, name=None):
        super(Patches, self).__init__(name=name)
        self.patch_width = patch_width
        self.patch_height = patch_height

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_width, self.patch_height, 1],
            strides=[1, self.patch_width, self.patch_height, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, name=None):
        super(PatchEncoder, self).__init__(name=name)
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class PatchDecoder(Layer):
    def __init__(self, patch_width, patch_height, x_patches, y_patches, projection_dim, mlp_units, name=None):
        super(PatchDecoder, self).__init__(name=name)
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.x_patches = x_patches
        self.y_patches = y_patches

        self.positional_embedding = Embedding(
            input_dim=self.x_patches * self.y_patches, output_dim=projection_dim
        )

        self.mlp = MLP([mlp_units, 2 * mlp_units, patch_width * patch_height], 0.1, activation='linear', name=f"{name}_mlp")
        # self.reshape = Reshape(target_shape=[self.patch_width, self.patch_height, 1],
        #                        input_shape=[1, self.patch_width * self.patch_height],
        #                        name=f"{name}_reshape_1")
        # self.concatenate = Concatenate(axis=-1)

    def call(self, encoded):
        positions = tf.range(start=0, limit=self.x_patches * self.y_patches, delta=1)
        embedding = self.positional_embedding(positions)

        # Extracting patches
        patches = self.mlp(encoded + embedding)
        reshaped = tf.reshape(patches, (-1, 16, 16, 16, 16))
        reshaped = tf.einsum("npqhw->nphqw", reshaped)
        reshaped = tf.reshape(reshaped, (-1, 256, 256, 1))

        # Merging into final output
        # def func(x):
        #     x = tf.nn.space_to_depth(x, self.patch_width)
        #     x = tf.reshape(x, [self.x_patches * self.patch_width, self.y_patches * self.patch_height, 1])
        #     return x

        return reshaped
