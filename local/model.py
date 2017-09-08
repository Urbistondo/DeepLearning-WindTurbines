import os

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_optimizer(learning_rate):
    optimizer = keras.optimizers.Adam(lr = learning_rate, beta_1 = 0.9,
                                      beta_2 = 0.999, epsilon = 1e-08,
                                      decay = 0.0)
    # optimizer = keras.optimizers.SGD(
    #     lr = learning_rate, momentum = 0.0, decay = 0.0, nesterov = False
    # )
    return optimizer


def model(layers, learning_rate):
    model = Sequential()
    first = True
    for layer in layers:
        if first:
            model.add(
                LSTM(
                    layer, input_shape = (1, 52), return_sequences = True,
                    kernel_initializer = 'he_normal'
                )
            )
            first = False
        else:
            model.add(
                LSTM(
                    layer, return_sequences = True,
                    kernel_initializer = 'he_normal'
                )
            )
    model.add(
        Dense(1, activation = 'relu', kernel_initializer = 'he_normal')
    )
    model.compile(
        loss = 'mean_squared_error', optimizer = get_optimizer(learning_rate)
    )
    return model


def peak_model(layers, learning_rate):
    model = Sequential()
    first = True
    for layer in layers:
        if first:
            model.add(
                LSTM(
                    layer, batch_input_shape = (10, 1, 52),
                    return_sequences = True, kernel_initializer = 'he_normal',
                    stateful = True
                )
            )
            first = False
        else:
            model.add(
                LSTM(
                    layer, return_sequences = True,
                    kernel_initializer = 'he_normal', stateful = True
                )
            )
    model.add(
        Dense(1, activation = 'relu', kernel_initializer = 'he_normal')
    )
    model.compile(
        loss = 'mean_squared_error', optimizer = get_optimizer(learning_rate)
    )
    return model


def fit_model(model, train_x, train_y, epochs, batch):
    history = model.fit(
        train_x, train_y, validation_split = 0.33, epochs = epochs,
        batch_size = batch, verbose = 0, shuffle = True,
    )
    return model, history


def save_model(model, output_dir, job_name, file_name, target, run = -1):
    directory = '%s/%d' % (output_dir, job_name)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    directory = '%s/%s' % (directory, file_name)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    directory = '%s/%s' % (directory, target)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    if run > -1:
        directory = '%s/%d' % (directory, run)
        os.mkdir(directory)
    with open('%s/structure.json' % directory, mode = 'w') as output_file:
        output_file.write(model.to_json())
        model.save_weights('%s/weights.h5' % directory)
    print('Model successfully saved.')


def load_model(model, file_name):
    with open(file_name, mode = 'r') as input_file:
        model.load_weights(input_file.name)
    print('Weights successfully loaded.')
    return model
