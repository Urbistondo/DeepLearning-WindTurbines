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
        train_files, predict_files, output_dir, targets, batch_size, model_size,
        epochs, learning_rate, kfold_splits, *layers
):
    """Trains the Recurrent Neural Network.

        It performs all the necessary preparations such as:
         - Reading input data from a CSV
         - Normalizing data
         - Reshaping data
         - Spliting data into train and test sets
         - Training the model performing KFold validation
         - Predicting values
         - Plotting results and saving them

        Args:
        @:type  train_file: string
        @:param train_file: Path to the input CSV file with the training data.
        @:type  predict_file: string
        @:param predict_file: Path to the input CSV file with the predicting
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
        @:type  model_size: string
        @:param model_size: Indicates if the model receives few variables
        ('small') or if it receives lots of them ('big')
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

        """

    scaler = MinMaxScaler(feature_range = (0, 1))
    for directory in os.walk(output_dir):
        subdirectories = np.array(directory[1]).astype(int)
        break
    job_name = np.max(subdirectories) + 1

    # TODO Create copy of this training method, substituting kfold for manual
    # TODO in order to train on a single aero's file.
    for train_file, predict_file in zip(train_files, predict_files):
        for target in targets:
            x, y = dp.get_data(train_file, 1, model_size, target)
            x, y = dp.normalize_data(x, y, scaler)

            kfold = StratifiedKFold(
                y, n_folds = kfold_splits, shuffle = False, random_state = seed
            )

            temp_history_loss = []
            temp_history_val_loss = []

            ml_model = mdl.model(layers[0], learning_rate)
            initial_time = t.time()
            for train_index, test_index in kfold:
                x, y = dp.reshape_data(x, y)
                ml_model, history = mdl.fit_model(
                    ml_model, x[train_index], y[train_index], epochs, batch_size
                )
                temp_history_loss.append(history.history['loss'])
                temp_history_val_loss.append(history.history['val_loss'])
            time = t.time() - initial_time

            acc_history_loss = []
            for history in temp_history_loss:
                for element in history:
                    acc_history_loss.append(element)
            acc_history_val_loss = []
            for history in temp_history_val_loss:
                for element in history:
                    acc_history_val_loss.append(element)

            file_name = train_file[-9:-4]
            current_model = {'batch': batch_size,
                             'epochs': epochs,
                             # 'train_results': final_train_scores,
                             # 'test_results': final_test_scores,
                             'time': time,
                             'layers': layers[0],
                             'learning_rate': learning_rate,
                             }

            mdl.save_model(ml_model, output_dir, job_name, file_name, target)
            results, y, predictions = pred.predict(
                ml_model, predict_file, model_size, batch_size, target
            )
            datetimes = np.asarray(dp.get_column(predict_file, 'unixtime'))
            real_outputs = np.asarray(dp.get_column(predict_file, target))
            lh.log_performance(
                current_model, results, datetimes, real_outputs, predictions,
                acc_history_loss, acc_history_val_loss, output_dir, job_name,
                file_name, target
            )
            lh.create_plotters(output_dir, job_name, file_name, target)
            ph.show_plot(y, predictions, target, 'Model predictions')
            del ml_model


def train_models(train_file, predict_file, output_dir, targets, model_size):
    """Main method that trains the Neural Network.

        It performs all the necessary preparations such as acquiring the data,
        normalizing and reshaping it, dividing it into train and test sets
        and training the model performing KFold validation, and saves the results.

        @:type  train_file: string
        @:param train_file: Path to the input CSV file.
        @:type  output_dir: string
        @:param output_dir: Path where the training output is saved.
        @:type  job_name: string
        @:param job_name: Name of the job being run.
        @:type  hp_tuning: boolean
        @:param hp_tuning: Indicates whether the current run is a normal training
         run (True) or a hyperparameter tuning run (False).
        @:type  batch_size: integer
        @:param batch_size: Indicates the size of the batches that are fed to the
         Neural Network.
        @:type  epochs: integer
        @:param epochs: Indicates the number of epochs for the training.
        @:type  layer1_width: integer
        @:param layer1_width: Number of neurons in the first layer.
        @:type  layer2_width: integer
        @:param layer2_width: Number of neurons in the second layer.
        @:type  learning_rate: float
        @:param learning_rate: Indicates the learning rate for the training
        """

    initial_time = t.time()

    list_kfolds = [3]
    list_epochs = [50, 100, 200]
    list_batches = [1000]
    list_widths = [2, 4, 8, 16, 32, 64, 128]
    list_layers = [2, 3, 4, 5, 6]
    list_rates = [0.0005]
    perms = []
    # for l in list_layers:
    #     for p in permutations(list_widths, l):
    #         perms.append(list(p))
    perms.append([128])
    best_model = {'batch': np.nan,
                  'epochs': np.nan,
                  'job_name': np.nan,
                  'train_results': np.nan,
                  'test_results': np.nan,
                  'time': np.nan,
                  'layers': np.nan,
                  }
    # best_ml_model = None
    for directory in os.walk(output_dir):
        subdirectories = np.array(directory[1]).astype(int)
        break
    job_name = np.max(subdirectories) + 1
    run = 1
    best_run = -1
    scaler = MinMaxScaler(feature_range = (0, 1))
    for epochs in list_epochs:
        for batch_size in list_batches:
            for learning_rate in list_rates:
                for kfold_split in list_kfolds:
                    for permutation in perms:
                        for target in targets:
                            # scaler = MinMaxScaler(feature_range = (0, 1))
                            x, y = dp.get_data(
                                train_file, 1, model_size, target
                            )
                            x, y = dp.normalize_data(x, y, scaler)

                            kfold = StratifiedKFold(
                                y, n_folds = kfold_split, shuffle = True,
                                random_state = seed
                            )

                            final_train_scores = []
                            final_test_scores = []
                            temp_history_loss = []
                            temp_history_val_loss = []

                            ml_model = mdl.model(
                                permutation, learning_rate)
                            for train_index, test_index in kfold:
                                x, y = dp.reshape_data(x, y)
                                ml_model, history = mdl.fit_model(
                                    ml_model, x[train_index], y[train_index],
                                    epochs, batch_size)
                                temp_history_loss.append(
                                    history.history['loss'])
                                temp_history_val_loss.append(
                                    history.history['val_loss'])
                                final_train_scores.append(
                                    ml_model.evaluate(x[train_index],
                                                      y[train_index]))
                                final_test_scores.append(
                                    ml_model.evaluate(x[train_index],
                                                      y[train_index]))

                            acc_history_loss = []
                            for history in temp_history_loss:
                                for element in history:
                                    acc_history_loss.append(element)
                            acc_history_val_loss = []
                            for history in temp_history_val_loss:
                                for element in history:
                                    acc_history_val_loss.append(element)

                            time = t.time() - initial_time
                            file_name = train_file[-9:-4]
                            current_model = {'batch': batch_size,
                                             'epochs': epochs,
                                             'train_results': final_train_scores,
                                             'test_results': final_test_scores,
                                             'time': time,
                                             'layers': permutation,
                                             'learning_rate': learning_rate,
                                             }

                            mdl.save_model(
                                ml_model, output_dir, job_name, file_name,
                                target, run
                            )
                            results, y, predictions = pred.predict(
                                ml_model, predict_file, model_size, batch_size,
                                target
                            )
                            datetimes = np.asarray(
                                dp.get_column(predict_file, 'unixtime')
                            )
                            real_outputs = np.asarray(
                                dp.get_column(predict_file, target)
                            )
                            lh.log_performance(
                                current_model, results, datetimes, real_outputs,
                                predictions, output_dir, job_name, file_name,
                                target, run
                            )
                            ph.save_plot(
                                output_dir, job_name, file_name, target,
                                'predictions', y, predictions, run = run
                            )
                            ph.save_plot(
                                output_dir, job_name, file_name, target,
                                'model_loss', acc_history_loss,
                                acc_history_val_loss, run = run
                            )
                            if (run == 1) | (np.mean(best_model['test_results'])
                                                 > np.mean(final_test_scores)):
                                # best_ml_model = ml_model
                                best_model = current_model
                                best_run = run
                            run += 1
                            del ml_model
        print('BEST MODEL: %d' % best_run)
        initial_time = t.time()


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
        '--model-size',
        help = 'Big model or small model',
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
            int(arguments['batch_size']), arguments['model_size'],
            int(arguments['epochs']), float(arguments['learning_rate']),
            int(arguments['kfold_splits']), layers
        )
    # Train various models with different topologies and sets of hyperparameters
    else:
        train_models(arguments['train_file'], arguments['predict_file'],
                     arguments['job_dir'], arguments['targets'],
                     arguments['model_size'])
