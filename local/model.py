import os

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_optimizer(learning_rate):
    optimizer = keras.optimizers.Adam(lr = learning_rate, beta_1 = 0.9,
                                      beta_2 = 0.999, epsilon = 1e-08,
                                      decay = 0.0)
    return optimizer


def normal_model(layers, learning_rate):
    model = Sequential()
    first = True
    for layer in layers:
        if first:
            model.add(LSTM(layer, input_shape = (1, 52),
                           return_sequences = True,
                           kernel_initializer = 'he_normal'))
            first = False
        else:
            model.add(LSTM(layer, return_sequences = True,
                           kernel_initializer = 'he_normal'))
    model.add(Dense(1, activation = 'relu',
                    kernel_initializer = 'he_normal'))
    model.compile(loss = 'mean_squared_error',
                  optimizer = get_optimizer(learning_rate))
    return model


def peak_model(layers, learning_rate):
    model = Sequential()
    first = True
    for layer in layers:
        if first:
            model.add(LSTM(layer, input_shape = (1, 52),
                           return_sequences = True,
                           kernel_initializer = 'he_normal'))
            first = False
        else:
            model.add(LSTM(layer, return_sequences = True,
                           kernel_initializer = 'he_normal'))
    model.add(Dense(1, activation = 'relu',
                    kernel_initializer = 'he_normal'))
    model.compile(loss = 'mean_squared_error',
                  optimizer = get_optimizer(learning_rate))
    return model


def fit_model(model, train_x, train_y, epochs, batch):
    history = model.fit(train_x, train_y, validation_split = 0.33,
                        epochs = epochs, batch_size = batch, verbose = 0)
    return model, history


def save_model(model, output_dir, job_name, file_name):
    if not os.path.isdir('%s/%d' % (output_dir, job_name)):
        os.mkdir('%s/%d' % (output_dir, job_name))
    if not os.path.isdir('%s/%d/%s' % (output_dir, job_name, file_name)):
        os.mkdir('%s/%d/%s' % (output_dir, job_name, file_name))
    with open('%s/%d/%s/structure.json' % (output_dir, job_name, file_name),
              mode = 'w') as output_file:
        output_file.write(model.to_json())
        model.save_weights('%s/%d/%s/weights.h5' % (output_dir, job_name,
                                                    file_name))
    print('Model successfully saved.')


def load_model(ml_model, file_name):
    with open(file_name, mode = 'r') as input_file:
        ml_model.load_weights(input_file.name)
    print('Weights successfully loaded.')
    return ml_model
