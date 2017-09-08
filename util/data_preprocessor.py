import numpy as np
import pandas as pd

from util import preprocessor as pp
from tensorflow.python.lib.io.file_io import FileIO


'''Takes two tensors and if they're not in 3D it reshapes them into 3D tensors
   to feed to the LSTM layers of the network'''


def reshape_data(x, y):
    # If data is not 3-dimensional, reshape it to 3 dimensions
    if len(x.shape) != 3:
        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        y = np.reshape(y, (y.shape[0], 1, 1))
    return x, y


'''Special method for reading data from CSV file when running from Google
   Cloud Platform. It uses TensorFlow's FileIO instead of Pandas' read_csv'''


def read_with_tf(file_path):
    # Create empty pandas dataframe with desired shape to store entries from CSV
    # file
    df = pd.DataFrame(np.zeros(52).reshape(1, 52))
    counter = 0
    with FileIO(file_path, 'r') as f:
        for line in f:
            if line:
                # Create header from the CSV's first line
                if counter == 0:
                    columns = line.split(',')
                    columns[len(columns) - 1] = columns[len(columns) - 1][:-1]
                    df.columns = columns
                # For every other entry, copy it into the dataframe
                else:
                    new_line = line.split(',')
                    new_line[len(new_line) - 1] = \
                        new_line[len(new_line) - 1][:-1]
                    df.loc[counter] = new_line
                counter += 1
    return df


'''Get input and output data from a file specifying the name of the target
   variable'''


def get_data(file_path, look_back, target):
    # If being run from Google Cloud Platform, use especial method for obtaining
    # dataframe from CSV
    if file_path.split('/')[2] == 'nemsolutions-gcp-databucket':
        final_data = read_with_tf(file_path)
    # Otherwise, if being run locally, use default method
    else:
        final_data = pp.read_data(file_path)
    x = final_data
    # Drop target variable from input dataframe
    if target == 'WGENBearNDETemp':
        x = x.drop(x.columns[15], axis = 1)
    elif target == 'WGENBearDETemp':
        x = x.drop(x.columns[22], axis = 1)
    y = final_data[target]
    return x.values, y.values


'''Normalize data using provided scaler'''


def normalize_data(x, y, scaler):
    return scaler.fit_transform(x), scaler.fit_transform(y)


'''Denormalize data using provided scaler'''


def denormalize_data(normalized_y, scaler):
    return scaler.inverse_transform(normalized_y)


'''Get column by name from CSV file'''


def get_column(file_path, column_name):
    # If being run from Google Cloud Platform, use especial method
    if file_path.split('/')[2] == 'nemsolutions-gcp-databucket':
        data = read_with_tf(file_path)
    # Otherwise, if being run locally, use default method
    else:
        data = pp.read_data(file_path)
    return data[column_name]

