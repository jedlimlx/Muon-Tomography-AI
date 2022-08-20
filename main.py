from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from layers import ResidualStack

input = Input(shape=(64, 64, 3))
x = ResidualStack(64, 3, name="stack_1")(input)

model = Model(inputs=input, outputs=x)
model.compile(optimizer="adam", loss="mse")
model.summary()
