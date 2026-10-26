import numpy as np
from machine_learning.layer import Layer


class Algorithm:
    def __init__(self):
        self.name = None
        self.layer_list = []

    def set_algorithm_data(self, model):
        if self:
            pass
        import tensorflow as tf
        from keras_flops import get_flops

        self.name = model.name

        flops_size = 0
        inp = np.ones((224, 224, 3))
        for i in range(0, len(model.layers)):
            new_model = tf.keras.models.Model(inputs=[model.input], outputs=[model.layers[i].output])
            layer = Layer()
            layer.name = model.layers[i].name
            if type(model.layers[i].output_shape) is tuple:
                output_shape = list(model.layers[i].output_shape)
            else:
                output_shape = list(model.layers[i].output_shape[0])
            output_shape = [i for i in output_shape if i is not None]
            layer.data_size = np.prod(np.array(output_shape))
            layer.flops_size = get_flops(new_model, batch_size=1) - flops_size
            flops_size = flops_size + layer.flops_size
            self.layer_list.append(layer)

"""
algo = Algorithm()

import keras
from keras_flops import get_flops
from keras import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, MaxPooling2D
from keras.layers import Flatten, Conv2D

# build model
inp = Input((32, 32, 3))
x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
x = Conv2D(65, (3, 3), activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(10, activation="softmax")(x)
model = Model(inp, out)


model = keras.applications.VGG16(include_top=True,
                                 weights=None,
                                 input_tensor=None,
                                 input_shape=(127, 127, 3),
                                 pooling='max',
                                 classes=2)

w = get_flops(model, batch_size=1)
print(f"FLOPS: {w / 10 ** 9:.03} G")


algo.set_algorithm_data(model)

print(algo)
"""