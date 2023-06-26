from keras.layers import *
from keras.models import *
from typing import *
from convnext_block import ConvNeXtBlock

_mlp_base_params = {
    'units': []
}


class MLP(Layer):
    def __init__(self, units, activations, **kwargs):
        super(MLP, self).__init__(**kwargs)

        assert len(units) == len(activations)

        self.units = units
        self.activations = activations

        self.layers = [
            Dense(units=units[i], activation=activations[i], name=f'{self.name}/dense_{i}')
            for i in range(len(units))
        ]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
        cfg = super(MLP, self).get_config()
        cfg.update({
            'units': self.units,
            'activations': self.activations
        })


class Agg2D(Model):
    def __init__(
            self,
            mlp_units=None,
            mlp_activations=None,
            downward_blocks=None,
            downward_filters=None,
            upward_blocks=None,
            upward_filters=None,
            *args, **kwargs):
        super(Agg2D, self).__init__(*args, **kwargs)

        if type(mlp_activations) is str:
            mlp_activations = [mlp_activations] * len(mlp_units)

        self.mlp = MLP(mlp_units, mlp_activations)

        self.downward_convs = []
        self.downsampling = []
        self.upward_convs = []
        self.upsampling = []
        for stage in range(len(downward_blocks)):
            stack = []
            for c in range(downward_blocks[stage]):
                stack.append(ConvNeXtBlock(projection_dim=downward_filters[stage],
                                           dims=2,
                                           name=f'{self.name}/stage_{stage}/block_{c}'))
                self.downward_convs.append(Sequential(stack))

        for stage in range(len(downward_blocks) - 1):
            self.downsampling.append(Sequential([
                LayerNormalization(epsilon=1e-6, name=f'{self.name}/stage_{stage}/downsampling/layer_norm'),
                Conv2D(filters=downward_filters[stage + 1],
                       kernel_size=2,
                       strides=2,
                       name=f'{self.name}/stage_{stage}/downsampling/conv2d',
                       padding='same')
            ], name=f'{self.name}/stage_{stage}/downsampling'))

            stack = []
            for c in range(upward_blocks[stage]):
                stack.append(ConvNeXtBlock(projection_dim=upward_filters[stage],
                                           dims=2,
                                           name=f'{self.name}/up_{stage}/block_{c}'))
            self.upward_convs.append(Sequential(stack))

            self.upsampling.append(Sequential([
                LayerNormalization(epsilon=1e-6, name=f'{self.name}/up_{stage}/upsampling/layer_norm'),
                UpSampling2D(size=(2, 2), name=f'{self.name}/up_{stage}/upsampling/upsampling'),
                Conv2D(filters=upward_filters[stage],
                       kernel_size=3,
                       padding='same',
                       name=f'{self.name}/up_{stage}/upsampling/conv2d')
            ], name=f'{self.name}/up_{stage}/upsampling'))

    def call(self, inputs, training=None, mask=None):
        x = self.mlp(inputs)
        return x



