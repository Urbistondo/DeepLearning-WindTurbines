import numpy as np
import pandas as pd

from util import preprocessor
from tensorflow.python.lib.io.file_io import FileIO

# Takes two tensors and if they're not in 3D it reshapes them into 3D tensors to
# feed to the LSTM layers of the network
def reshape_data(x, y):
    if len(x.shape) != 3:
        reshaped_x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        reshaped_y = np.reshape(y, (y.shape[0], 1, 1))
    else:
        reshaped_x = x
        reshaped_y = y
    return reshaped_x, reshaped_y


def read_with_tf(file_path):
    df = pd.DataFrame(np.zeros(49).reshape(1, 49))
    counter = 0
    with FileIO(file_path, 'r') as f:
        for line in f:
            if line:
                if counter == 0:
                    columns = line.split(',')
                    columns[len(columns) - 1] = columns[len(columns) - 1][:-1]
                    df.columns = columns
                else:
                    new_line = line.split(',')
                    new_line[len(new_line) - 1] = \
                        new_line[len(new_line) - 1][:-1]
                    df.loc[counter] = new_line
                counter += 1
    return df


def get_data(file_path, look_back, model_size, target):
    if file_path.split('/')[2] == 'nemsolutions-gcp-databucket':
        final_data = read_with_tf(file_path)
    else:
        final_data = preprocessor.read_data(file_path)
    x = final_data
    if target == 'WGENBearNDETemp':
        x = x.drop(x.columns[15], axis = 1)
    elif target == 'WGENBearDETemp':
        x = x.drop(x.columns[22], axis = 1)
    y = final_data[target]
    return x.values, y.values


def normalize_data(x, y, scaler):
    return scaler.fit_transform(x), scaler.fit_transform(y)


def denormalize_data(y, normalized_y, scaler):
    return scaler.inverse_transform(normalized_y)


def get_column(file_path, column_name):
    if file_path.split('/')[2] == 'nemsolutions-gcp-databucket':
        data = read_with_tf(file_path)
    else:
        data = preprocessor.read_data(file_path)
    return data[column_name]

