from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers import ResidualStack

blocks = [6, 12, 24, 16, 12]

input = Input(shape=(256, 256, 3))
x = ResidualStack(64, blocks[0], name="conv1", activation="swish")(input)
x = ResidualStack(64, blocks[1], name="conv2", activation="swish")(x)

model = Model(inputs=input, outputs=x)
model.compile(optimizer="adam", loss="mse")
model.summary()
