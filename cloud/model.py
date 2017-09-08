import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.python.lib.io.file_io import FileIO


'''Returns an optimizer with the specified learning rate'''


def get_optimizer(learning_rate):
    optimizer = keras.optimizers.Adam(
        lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08,
        decay = 0.0
    )
    return optimizer


'''Builds an LSTM model with the topology specified in the list "layers" and
   compiles it with an optimizer'''


def model(layers, learning_rate):
    model = Sequential()
    first = True
    for l in layers:
        if first:
            model.add(
                LSTM(
                    l, input_shape = (1, 52), return_sequences = True,
                    kernel_initializer = 'he_normal'
                )
            )
            first = False
        else:
            model.add(
                LSTM(
                    l, return_sequences = True,
                    kernel_initializer = 'he_normal')
            )
    model.add(
        Dense(
            1, activation = 'relu', kernel_initializer = 'normal')
    )
    model.compile(
        loss = 'mean_squared_error', optimizer = get_optimizer(learning_rate)
    )
    return model


'''Fits the model to the provided input and target variable'''


def fit_model(model, train_x, train_y, epochs, batch, callbacks):
    history = model.fit(
        train_x, train_y, validation_split = 0.33, epochs = epochs,
        batch_size = batch, verbose = 0, callbacks = [callbacks])
    return model, history


'''Save a trained model's weights in .h5 format and model structure to JSON'''


def save_model(model, output_dir, file_name, target, run = -1):
    directory = '%s/%s/%s' % (output_dir, file_name, target)
    # Identifies whether multiple trainings for the same train-predict files
    # are being run in order to create separate output directories for each
    if run > -1:
        directory = '%s/%d' % (directory, run)
    with FileIO('%s/structure.json' % directory, mode = 'w') as output_file:
        output_file.write(model.to_json())
    model.save_weights('weights.h5')
