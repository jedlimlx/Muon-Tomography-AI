from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers import DenseBlock, TransitionBlock

blocks = [6, 12, 24, 16, 12]

input = Input(shape=(256, 256, 3))
x = DenseBlock(blocks[0], name="conv1")(input)
x = TransitionBlock(0.5, name="pool1")(x)
x = DenseBlock(blocks[1], name="conv2")(x)
x = TransitionBlock(0.5, name="pool2")(x)
x = DenseBlock(blocks[2], name="conv3")(x)
x = TransitionBlock(0.5, name="pool3")(x)
x = DenseBlock(blocks[3], name="conv4")(x)
x = TransitionBlock(0.5, name="pool4")(x)
x = DenseBlock(blocks[4], name="conv5")(x)

model = Model(inputs=input, outputs=x)
model.compile(optimizer="adam", loss="mse")
model.summary()
