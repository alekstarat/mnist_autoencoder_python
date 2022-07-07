from keras.datasets import mnist
import numpy as np


def data_preparation():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.array(x_train)
    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

    return x_train

