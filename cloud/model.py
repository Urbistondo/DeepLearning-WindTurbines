import os

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.file_io import FileIO
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


def optimizer(learning_rate):
    optimizer = keras.optimizers.Adam(lr = learning_rate, beta_1 = 0.9,
                                      beta_2 = 0.999, epsilon = 1e-08,
                                      decay = 0.0)
    return optimizer


def model(layers, learning_rate):
    model = Sequential()
    first = True
    for l in layers:
        if first:
            model.add(LSTM(l, input_shape = (1,3), return_sequences = True,
                           kernel_initializer = 'he_normal'))
            # model.add(Dropout(0.05))
            first = False
        else:
            model.add(LSTM(l, return_sequences = True,
                           kernel_initializer = 'he_normal'))
            # model.add(Dropout(0.05))
    model.add(Dense(1, activation = 'relu',
                    kernel_initializer = 'normal'))
    model.compile(loss = 'mean_squared_error',
                  optimizer = optimizer(learning_rate))
    return model


def big_model(layers, learning_rate = 0.1):
    model = Sequential()
    first = True
    for l in layers:
        if first:
            model.add(LSTM(l, input_shape = (1,48),
                           return_sequences = True,
                           kernel_initializer = 'he_normal'))
            first = False
        else:
            model.add(LSTM(l, return_sequences = True,
                           kernel_initializer = 'he_normal'))
    model.add(Dense(1, activation = 'relu',
                    kernel_initializer = 'normal'))
    model.compile(loss = 'mean_squared_error',
                  optimizer = optimizer(learning_rate))
    return model


def fit_model(model, train_x, train_y, epochs, batch, callbacks):
    history = model.fit(train_x, train_y, validation_split = 0.33,
                        epochs = epochs, batch_size = batch, verbose = 0,
                        callbacks = [callbacks])
    return model, history


def save_model(model, output_dir, file_name):
    with FileIO('%s/%s/structure.json' %
                        (output_dir, file_name),
                        mode = 'w') as output_file:
        output_file.write(model.to_json())
    model.save_weights('weights.h5')