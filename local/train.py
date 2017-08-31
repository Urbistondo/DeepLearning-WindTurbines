import argparse
import os
import time as t

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from itertools import permutations

from local import model as mdl
from local import predict as pred
from util import log_handler as lh
from util import plot_handler as ph
from util import data_preprocessor as dp


desired_width = 180
pd.set_option('display.width', desired_width)


def train_custom_model(train_file, predict_file, output_dir, targets, batch_size,
                       model_size, epochs, learning_rate, kfold_splits,
                       *layers):
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

    scaler = MinMaxScaler(feature_range = (0, 1))

    
    x, y = dp.get_data(train_file, 1, model_size)
    x, y = dp.normalize_data(x, y, scaler)

    kfold = StratifiedKFold(y, n_folds = kfold_splits, shuffle = True,
                            random_state = seed)

    final_train_scores = []
    final_test_scores = []
    temp_history_loss = []
    temp_history_val_loss = []
    for directory in os.walk(output_dir):
        subdirectories = np.array(directory[1]).astype(int)
        break
    job_name = np.max(subdirectories) + 1
    ml_model = mdl.normal_model(layers[0], learning_rate)
    for train_index, test_index in kfold:
        x, y = dp.reshape_data(x, y)
        ml_model, history = mdl.fit_model(ml_model, x[train_index],
                                          y[train_index], epochs,
                                          batch_size)
        temp_history_loss.append(history.history['loss'])
        temp_history_val_loss.append(history.history['val_loss'])
        final_train_scores.append(ml_model.evaluate(x[train_index],
                                                    y[train_index]))
        final_test_scores.append(ml_model.evaluate(x[test_index],
                                                   y[test_index]))

    acc_history_loss = []
    for history in temp_history_loss:
        for element in history:
            acc_history_loss.append(element)
    acc_history_val_loss = []
    for history in temp_history_val_loss:
        for element in history:
            acc_history_val_loss.append(element)

    time = t.time() - initial_time
    current_model = {'batch': batch_size,
                     'epochs': epochs,
                     'train_results': final_train_scores,
                     'test_results': final_test_scores,
                     'time': time,
                     'layers': layers[0],
                     'learning_rate': learning_rate,
                     }
    file_name = train_file[-9:-4]

    mdl.save_model(ml_model, output_dir, job_name, file_name)
    # del ml_model
    # if model_size == 'small':
    #     ml_model2 = mdl.small_model(layers[0], learning_rate)
    # else:
    #     ml_model2 = mdl.big_model(layers[0], learning_rate)
    # mdl.load_model(ml_model, '%s/%s/%s/%s' % (output_dir, job_name, file_name,
    #  'weights.h5'))
    results, y, predictions = pred.normal_predict(ml_model, predict_file, model_size,
                                           batch_size)
    datetimes = np.asarray(dp.get_column(predict_file, 'unixtime'))
    nde_temp = np.asarray(dp.get_column(predict_file, target))
    lh.log_performance(current_model, results, datetimes, nde_temp,
                       predictions, output_dir, job_name, file_name, target)
    ph.show_plot(y, predictions)
    ph.save_plot(output_dir, job_name, file_name, 'predictions',
                 y, predictions)
    ph.save_plot(output_dir, job_name, file_name, 'model_loss',
                 acc_history_loss, acc_history_val_loss)
    del ml_model


def train_models(train_file, predict_file, output_dir, model_size):
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

    list_kfolds = [3, 5, 10]
    list_epochs = [50, 100, 200]
    list_batches = [500, 1000, 2500, 5000]
    list_widths = [2, 4, 8, 16, 32, 64, 128]
    list_layers = [2, 3, 4, 5, 6]
    list_rates = [0.1, 0.01, 0.001, 0.0001]
    perms = []
    for l in list_layers:
        for p in permutations(list_widths, l):
            perms.append(list(p))
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
    for epochs in list_epochs:
        for batch_size in list_batches:
            for learning_rate in list_rates:
                for kfold_split in list_kfolds:
                    for permutation in perms:
                        scaler = MinMaxScaler(feature_range = (0, 1))
                        x, y = dp.get_data(train_file, 1, model_size)
                        x, y = dp.normalize_data(x, y, scaler)

                        kfold = StratifiedKFold(y, n_folds = kfold_split,
                                                shuffle = True,
                                                random_state = seed)

                        final_train_scores = []
                        final_test_scores = []
                        temp_history_loss = []
                        temp_history_val_loss = []
                        if model_size == 'small':
                            ml_model = mdl.small_model(permutation, learning_rate)
                        else:
                            print('Permutation: ', permutation)
                            ml_model = mdl.normal_model(permutation, learning_rate)
                        for train_index, test_index in kfold:
                            x, y = dp.reshape_data(x, y)
                            ml_model, history = mdl.fit_model(ml_model, x[train_index],
                                                              y[train_index], epochs,
                                                              batch_size)
                            temp_history_loss.append(history.history['loss'])
                            temp_history_val_loss.append(history.history['val_loss'])
                            final_train_scores.append(ml_model.evaluate(x[train_index],
                                                                        y[train_index]))
                            final_test_scores.append(ml_model.evaluate(x[train_index],
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

                        mdl.save_model(ml_model, output_dir, job_name, file_name)
                        results, y, predictions = pred.predict(ml_model, predict_file, model_size)
                        datetimes = np.asarray(dp.get_column(predict_file, 'unixtime'))
                        real_power = np.asarray(dp.get_column(predict_file, 'WTURPower'))
                        lh.log_performance(current_model, results, datetimes, real_power,
                                           predictions, output_dir, job_name, file_name, run)
                        if (run == 1) | (np.mean(best_model['test_results'])
                                                 > np.mean(final_test_scores)):
                            # best_ml_model = ml_model
                            best_model = current_model
                        run += 1
                        del ml_model
        lh.log_performance(best_model, results, datetimes, real_power,
                           predictions, output_dir, job_name, file_name, run)
        initial_time = t.time()
    # mdl.save_model(best_ml_model, output_dir, job_name, file_name)
    # lh.log_results(best_model, output_dir, job_name, file_name, True)


if __name__ == '__main__':
    # Avoid unnecessary warning logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Set seed for reproducibility
    seed = 42
    np.random.seed(seed)
    tf.set_random_seed(seed)

    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help = 'GCS or local paths to training data',
        required = True
    )
    parser.add_argument(
        '--predict-file',
        help = 'GCS or local paths to testing data',
        required = True
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
    if arguments['custom'] == 'True':
        layers = [int(l) for l in arguments['layers']]
        train_custom_model(arguments['train_file'], arguments['predict_file'],
                           arguments['job_dir'], arguments['targets'],
                           int(arguments['batch_size']),
                           arguments['model_size'], int(arguments['epochs']),
                           float(arguments['learning_rate']),
                           int(arguments['kfold_splits']), layers)
    else:
        train_models(arguments['train_file'], arguments['predict_file'],
                     arguments['job_dir'], arguments['model_size'])
