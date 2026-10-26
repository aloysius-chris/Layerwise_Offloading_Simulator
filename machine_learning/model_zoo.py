"""The eight deep learning models used as tasks in the experiments
(Table 3-2 of the thesis), built with Keras in their standard inference
configuration (no pre-trained weights are needed: only the architecture
matters for the simulation).

Five come from keras.applications; LeNet-5, AlexNet and GoogLeNet
(Inception v1) are not part of keras.applications and are built here from
their original papers.
"""
import keras
from keras import layers


def lenet5(input_shape=(32, 32, 1), classes=10):
    """LeNet-5 (LeCun et al., 1998): C1-S2-C3-S4-C5-F6-output, tanh units."""
    x_in = layers.Input(input_shape, name="input")
    x = layers.Conv2D(6, 5, activation="tanh", name="c1")(x_in)
    x = layers.AveragePooling2D(2, name="s2")(x)
    x = layers.Conv2D(16, 5, activation="tanh", name="c3")(x)
    x = layers.AveragePooling2D(2, name="s4")(x)
    x = layers.Conv2D(120, 5, activation="tanh", name="c5")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(84, activation="tanh", name="f6")(x)
    x = layers.Dense(classes, activation="softmax", name="output")(x)
    return keras.Model(x_in, x, name="lenet5")


def alexnet(input_shape=(224, 224, 3), classes=1000):
    """AlexNet (Krizhevsky et al., 2012) in its standard single-tower form
    (the torchvision reference model, ~0.71 G multiply-adds): 5 conv + 3 fully
    connected layers.  The original two-GPU grouping and local response
    normalisation are not used in modern implementations."""
    x_in = layers.Input(input_shape, name="input")
    x = layers.ZeroPadding2D(2, name="pad1")(x_in)
    x = layers.Conv2D(64, 11, strides=4, activation="relu", name="conv1")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1")(x)
    x = layers.Conv2D(192, 5, padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool2")(x)
    x = layers.Conv2D(384, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv5")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool5")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.5, name="drop6")(x)
    x = layers.Dense(4096, activation="relu", name="fc6")(x)
    x = layers.Dropout(0.5, name="drop7")(x)
    x = layers.Dense(4096, activation="relu", name="fc7")(x)
    x = layers.Dense(classes, activation="softmax", name="fc8")(x)
    return keras.Model(x_in, x, name="alexnet")


def _inception(x, f1, f3r, f3, f5r, f5, fp, name):
    b1 = layers.Conv2D(f1, 1, padding="same", activation="relu", name=name + "_1x1")(x)
    b2 = layers.Conv2D(f3r, 1, padding="same", activation="relu", name=name + "_3x3_reduce")(x)
    b2 = layers.Conv2D(f3, 3, padding="same", activation="relu", name=name + "_3x3")(b2)
    b3 = layers.Conv2D(f5r, 1, padding="same", activation="relu", name=name + "_5x5_reduce")(x)
    b3 = layers.Conv2D(f5, 5, padding="same", activation="relu", name=name + "_5x5")(b3)
    b4 = layers.MaxPooling2D(3, strides=1, padding="same", name=name + "_pool")(x)
    b4 = layers.Conv2D(fp, 1, padding="same", activation="relu", name=name + "_pool_proj")(b4)
    return layers.Concatenate(name=name + "_output")([b1, b2, b3, b4])


def googlenet(input_shape=(224, 224, 3), classes=1000):
    """GoogLeNet / Inception v1 (Szegedy et al., 2015), inference graph: the
    two auxiliary classifiers (used only for training) and local response
    normalisation are omitted."""
    x_in = layers.Input(input_shape, name="input")
    x = layers.Conv2D(64, 7, strides=2, padding="same", activation="relu", name="conv1_7x7")(x_in)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)
    x = layers.Conv2D(64, 1, padding="same", activation="relu", name="conv2_3x3_reduce")(x)
    x = layers.Conv2D(192, 3, padding="same", activation="relu", name="conv2_3x3")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool2")(x)
    x = _inception(x, 64, 96, 128, 16, 32, 32, "inception_3a")
    x = _inception(x, 128, 128, 192, 32, 96, 64, "inception_3b")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool3")(x)
    x = _inception(x, 192, 96, 208, 16, 48, 64, "inception_4a")
    x = _inception(x, 160, 112, 224, 24, 64, 64, "inception_4b")
    x = _inception(x, 128, 128, 256, 24, 64, 64, "inception_4c")
    x = _inception(x, 112, 144, 288, 32, 64, 64, "inception_4d")
    x = _inception(x, 256, 160, 320, 32, 128, 128, "inception_4e")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool4")(x)
    x = _inception(x, 256, 160, 320, 32, 128, 128, "inception_5a")
    x = _inception(x, 384, 192, 384, 48, 128, 128, "inception_5b")
    x = layers.GlobalAveragePooling2D(name="pool5")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    x = layers.Dense(classes, activation="softmax", name="classifier")(x)
    return keras.Model(x_in, x, name="googlenet")


def vgg16():
    return keras.applications.VGG16(weights=None, input_shape=(224, 224, 3), classes=1000)


def resnet50():
    return keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3), classes=1000)


def densenet121():
    return keras.applications.DenseNet121(weights=None, input_shape=(224, 224, 3), classes=1000)


def mobilenetv2():
    return keras.applications.MobileNetV2(weights=None, input_shape=(224, 224, 3), classes=1000)


def efficientnetb0():
    return keras.applications.EfficientNetB0(weights=None, input_shape=(224, 224, 3), classes=1000)


# name used in the thesis -> builder
MODELS = {
    "LeNet-5": lenet5,
    "AlexNet": alexnet,
    "VGG-16": vgg16,
    "GoogleNet": googlenet,
    "ResNet-50": resnet50,
    "DenseNet-121": densenet121,
    "MobileNetV2": mobilenetv2,
    "EfficientNet-B0": efficientnetb0,
}
