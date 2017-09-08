import argparse
import os
import time as t

import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import permutations
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import train_test_split

from local import model as mdl
from local import predict as pred
from util import log_handler as lh
from util import plot_handler as ph
from util import data_preprocessor as dp


def train_model(
        train_files, predict_files, output_dir, targets, batch_size, epochs,
        learning_rate, kfold_splits, *layers
):
    '''

        Trains the Recurrent Neural Network.

        It performs all the necessary preparations such as:
         - Reading input data from a CSV
         - Normalizing data
         - Reshaping data
         - Spliting data into train and test sets
         - Training the model performing KFold validation
         - Predicting values
         - Plotting results and saving them

        Args:
        @:type  train_files: list
        @:param train_files: Paths to the input CSV files with the training
         data.
        @:type  predict_file: list
        @:param predict_file: Paths to the input CSV files with the predicting
         data.
        @:type  output_dir: string
        @:param output_dir: Path where the output is saved.
        @:type  targets: list
        @:param targets: List of variables to predict. A different model is
        created for each target variable provided. Currently supported:
            - WGENBearDETemp
            - WGENBearNDETemp
        @:type  batch_size: integer
        @:param batch_size: Indicates the size of the batches that are fed to
         the neural network.
        @:type  epochs: integer
        @:param epochs: Indicates the number of epochs for the training.
        @:type  learning_rate: float
        @:param learning_rate: Indicates the rate at which the model learns.
        @:type  kfold_splits: int
        @:param kfold_splits: Indicates the amount of splits applied to the
        training data when performing KFold validation.
        @:type  layers: list
        @:param layers: List of layers that determines the topology of the
        network.

    '''

    # Determine directory in which results of the training will be saved
    for directory in os.walk(output_dir):
        subdirectories = np.array(directory[1]).astype(int)
        break
    job_name = np.max(subdirectories) + 1
    # Obtain a scaler to normalize input data
    scaler = MinMaxScaler(feature_range = (0, 1))

    # Training and predicting for every pair of train-predict files
    for train_file, predict_file in zip(train_files, predict_files):
        # Training for a specific output variable
        for target in targets:
            # Read input data from training file
            x, y = dp.get_data(train_file, 1, target)
            # Normalize data using scaler
            x, y = dp.normalize_data(x, y, scaler)
            # Determine kfold to perform cross-validation
            kfold = StratifiedKFold(
                y, n_folds = kfold_splits, shuffle = False, random_state = seed
            )
            # Create temporary lists to store training and validation loss
            #  results
            temp_history_loss = []
            temp_history_val_loss = []
            # Obatin model with desired topology (determined by the list
            #  'layers[0]') and learning rate
            ml_model = mdl.model(layers[0], learning_rate)
            initial_time = t.time()
            for train_index, test_index in kfold:
                # Reshape data to obtain 3-dimensional data to feed LSTM layers
                x, y = dp.reshape_data(x, y)
                # Train the model
                ml_model, history = mdl.fit_model(
                    ml_model, x[train_index], y[train_index], epochs, batch_size
                )
                # Append loss data to temporary list
                temp_history_loss.append(history.history['loss'])
                temp_history_val_loss.append(history.history['val_loss'])
            time = t.time() - initial_time
            # Flatten training history from all K-Fold trainings into single
            # lists
            acc_history_loss = []
            for history in temp_history_loss:
                for element in history:
                    acc_history_loss.append(element)
            acc_history_val_loss = []
            for history in temp_history_val_loss:
                for element in history:
                    acc_history_val_loss.append(element)
            # Obtain name of the training file used
            file_name = train_file[-9:-4]
            # Save important parameters about trained model into dictionary
            current_model = {'batch': batch_size,
                             'epochs': epochs,
                             'time': time,
                             'layers': layers[0],
                             'learning_rate': learning_rate,
                             }
            # Save model weights
            mdl.save_model(ml_model, output_dir, job_name, file_name, target)
            # Use model to perform predictions on the data from the
            # corresponding predict file. It returns an abbreviated version of
            # the predicting performance in 'results' and the list of every
            # predicition performed
            results, predictions = pred.predict(
                ml_model, predict_file, target
            )
            # Extract "DATETIME" column from prediction data
            datetimes = np.asarray(dp.get_column(predict_file, 'unixtime'))
            # Extract real values of target variable from prediction data
            real_outputs = np.asarray(dp.get_column(predict_file, target))
            # Log the model's predicting performance
            lh.log_performance(
                current_model, results, datetimes, real_outputs, predictions,
                acc_history_loss, acc_history_val_loss, output_dir, job_name,
                file_name, target
            )
            # Create scripts that read loss and prediction results CSVs and plot
            # them
            lh.create_plotters(output_dir, job_name, file_name, target)
            # Show plot of the prediction results and the real values
            ph.show_plot(real_outputs, predictions, target, 'Model predictions')
            # Delete model
            del ml_model


def train_models(train_file, predict_file, output_dir, targets, model_size):
    '''

        Trains the Recurrent Neural Network.

        It performs all the necessary preparations such as:
         - Reading input data from a CSV
         - Normalizing data
         - Reshaping data
         - Spliting data into train and test sets
         - Training the model performing KFold validation
         - Predicting values
         - Plotting results and saving them

        Args:
        @:type  train_files: list
        @:param train_files: Paths to the input CSV files with the training
         data.
        @:type  predict_file: list
        @:param predict_file: Paths to the input CSV files with the predicting
         data.
        @:type  output_dir: string
        @:param output_dir: Path where the output is saved.
        @:type  targets: list
        @:param targets: List of variables to predict. A different model is
        created for each target variable provided. Currently supported:
            - WGENBearDETemp
            - WGENBearNDETemp
        @:type  batch_size: integer
        @:param batch_size: Indicates the size of the batches that are fed to
         the neural network.
        @:type  epochs: integer
        @:param epochs: Indicates the number of epochs for the training.
        @:type  learning_rate: float
        @:param learning_rate: Indicates the rate at which the model learns.
        @:type  kfold_splits: int
        @:param kfold_splits: Indicates the amount of splits applied to the
        training data when performing KFold validation.
        @:type  layers: list
        @:param layers: List of layers that determines the topology of the
        network.

        '''

    # List of hyperparameters to perform grid-search on. Note that a model will
    # be trained for every possible combination of hyperparameters and network
    # topology
    list_kfolds = [3]
    list_epochs = [50, 100, 200]
    list_batches = [1000]
    list_widths = [2, 4, 8, 16, 32, 64, 128]
    list_layers = [2, 3, 4, 5, 6]
    list_rates = [0.0005]
    # Create permutations of possible layers to create network topologies
    # The topology permutations don't take into account the
    # possibility of one or more layers being the same width.
    perms = []
    for l in list_layers:
        for p in permutations(list_widths, l):
            perms.append(list(p))
    # Determine directory in which results of the training will be saved
    for directory in os.walk(output_dir):
        subdirectories = np.array(directory[1]).astype(int)
        break
    job_name = np.max(subdirectories) + 1
    # Counter to keep track of how many models have been trained
    run = 1
    # Obtain a scaler to normalize input data
    scaler = MinMaxScaler(feature_range = (0, 1))

    # Perform one training for every combination of hyperparameters and network
    # topology
    for epochs in list_epochs:
        for batch_size in list_batches:
            for learning_rate in list_rates:
                for kfold_split in list_kfolds:
                    for permutation in perms:
                        for target in targets:
                            # Read input data from training file
                            x, y = dp.get_data(train_file, 1, target)
                            # Normalize data using scaler
                            x, y = dp.normalize_data(x, y, scaler)
                            # Determine kfold to perform cross-validation
                            kfold = StratifiedKFold(
                                y, n_folds = kfold_split, shuffle = True,
                                random_state = seed
                            )
                            # Create temporary lists to store training and validation loss
                            #  results
                            temp_history_loss = []
                            temp_history_val_loss = []
                            # Obatin model with desired topology (determined by the list
                            # 'permutation') and learning rate
                            ml_model = mdl.model(permutation, learning_rate)
                            initial_time = t.time()
                            for train_index, test_index in kfold:
                                # Reshape data to obtain 3-dimensional data to
                                # feed LSTM layers
                                x, y = dp.reshape_data(x, y)
                                # Train the model
                                ml_model, history = mdl.fit_model(
                                    ml_model, x[train_index], y[train_index],
                                    epochs, batch_size
                                )
                                temp_history_loss.append(
                                    history.history['loss']
                                )
                                temp_history_val_loss.append(
                                    history.history['val_loss']
                                )
                            time = t.time() - initial_time
                            # Flatten training history from all K-Fold trainings
                            # into single lists
                            acc_history_loss = []
                            for history in temp_history_loss:
                                for element in history:
                                    acc_history_loss.append(element)
                            acc_history_val_loss = []
                            for history in temp_history_val_loss:
                                for element in history:
                                    acc_history_val_loss.append(element)
                            # Obtain name of the training file used
                            file_name = train_file[-9:-4]
                            # Save important parameters about trained model into dictionary
                            current_model = {'batch': batch_size,
                                             'epochs': epochs,
                                             'time': time,
                                             'layers': permutation,
                                             'learning_rate': learning_rate,
                                             }
                            # Save model weights
                            mdl.save_model(
                                ml_model, output_dir, job_name, file_name,
                                target, run
                            )
                            # Use model to perform predictions on the data from
                            # the corresponding predict file. It returns an
                            # abbreviated version of the predicting performance
                            # in 'results' and the list of every predicition
                            # performed
                            results, predictions = pred.predict(
                                ml_model, predict_file, target
                            )
                            # Extract "DATETIME" column from prediction data
                            datetimes = np.asarray(
                                dp.get_column(predict_file, 'unixtime')
                            )
                            # Extract real values of target variable from
                            # prediction data
                            real_outputs = np.asarray(
                                dp.get_column(predict_file, target)
                            )
                            # Log the model's predicting performance
                            lh.log_performance(
                                current_model, results, datetimes, real_outputs,
                                predictions, acc_history_loss,
                                acc_history_val_loss, output_dir, job_name,
                                file_name, target, run
                            )
                            run += 1
                            # Create scripts that read loss and prediction
                            # results CSVs and plot them
                            lh.create_plotters(
                                output_dir, job_name, file_name, target
                            )
                            # Delete model
                            del ml_model


if __name__ == '__main__':
    # Avoid unnecessary warning logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    desired_width = 180
    pd.set_option('display.width', desired_width)

    # Set seed for reproducibility
    seed = 42
    np.random.seed(seed)
    tf.set_random_seed(seed)

    parser = argparse.ArgumentParser()
    # Input arguments
    parser.add_argument(
        '--train-files',
        help = 'GCS or local paths to training data',
        required = True,
        nargs = '+'
    )
    parser.add_argument(
        '--predict-files',
        help = 'GCS or local paths to testing data',
        required = True,
        nargs = '+'
    )
    parser.add_argument(
        '--job-dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--custom',
        help = 'Determines whether specific parameters are provided',
        required = True
    )
    parser.add_argument(
        '--targets',
        help = 'Specifies the target variable trying to be predicted',
        required = False,
        nargs = '+'
    )
    parser.add_argument(
        '--batch-size',
        help = 'Batch size',
        required = False
    )
    parser.add_argument(
        '--epochs',
        help = 'Number of epochs',
        required = False
    )
    parser.add_argument(
        '--learning-rate',
        help = 'Learning rate',
        required = False
    )
    parser.add_argument(
        '--kfold-splits',
        help = 'Number of splits for KFold cross-validation',
        required = False
    )
    parser.add_argument(
        '--layers',
        help = 'Widths of layers',
        required = False,
        nargs = '+'
    )
    args = parser.parse_args()
    arguments = args.__dict__
    # Train a model with a specific topology and set of hyperparametes
    if arguments['custom'] == 'True':
        layers = [int(l) for l in arguments['layers']]
        train_model(
            arguments['train_files'], arguments['predict_files'],
            arguments['job_dir'], arguments['targets'],
            int(arguments['batch_size']), int(arguments['epochs']),
            float(arguments['learning_rate']), int(arguments['kfold_splits']),
            layers
        )
    # Train various models with different topologies and sets of hyperparameters
    else:
        train_models(arguments['train_file'], arguments['predict_file'],
                     arguments['job_dir'], arguments['targets'],
                     arguments['model_size'])
