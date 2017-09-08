import os
import re
import argparse
import time as t

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from itertools import permutations

from cloud import model as mdl
from cloud import predict as pred
from util import log_handler as lh
from util import data_preprocessor as dp


def train_custom_model(
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

    # Obtain a scaler to normalize input data
    scaler = MinMaxScaler(feature_range = (0, 1))
    for train_file, predict_file in zip(train_files, predict_files):
        # Training for a specific output variable
        for target in targets:
            # Read input data from training file
            x, y = dp.get_data(train_file, 1, target)
            # Normalize data using scaler
            x, y = dp.normalize_data(x, y, scaler)
            # Determine kfold to perform cross-validation
            kfold = StratifiedKFold(
                y, n_folds = kfold_splits, shuffle = True, random_state = seed
            )
            # Create temporary lists to store training and validation loss
            #  results
            temp_history_loss = []
            temp_history_val_loss = []
            # Counter to keep track of how many models have been trained
            run = 1
            # Obatin model with desired topology (determined by the list
            #  'layers[0]') and learning rate
            ml_model = mdl.model(layers[0], learning_rate)
            initial_time = t.time()
            for train_index, test_index in kfold:
                # Reshape data to obtain 3-dimensional data to feed LSTM layers
                x, y = dp.reshape_data(x, y)
                # Create Keras Callback to plot training on TensorBoard
                tensorboard = TensorBoard(
                    log_dir = (
                        'gs://nemsolutions-gcp-databucket/output/%s/logs/%d'
                        % (output_dir.split('/')[-1], run)
                    ),
                    # histogram_freq = 0, batch_size = batch_size,
                    #  write_graph = True,
                    histogram_freq = 0, batch_size = batch_size,
                    write_graph = True, write_grads = False,
                    write_images = False, embeddings_freq = 0,
                    embeddings_layer_names = None,
                    embeddings_metadata = None
                )
                # Train the model
                ml_model, history = mdl.fit_model(
                    ml_model, x[train_index], y[train_index], epochs,
                    batch_size, tensorboard
                )
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
            mdl.save_model(ml_model, output_dir, run, file_name, target)
            # Use model to perform predictions on the data from the
            # corresponding predict file. It returns an abbreviated version of
            # the predicting performance in 'results' and the list of every
            # predicition performed
            results, predictions = pred.predict(ml_model, predict_file, target)
            # Extract "DATETIME" column from prediction data
            datetimes = np.asarray(dp.get_column(predict_file, 'unixtime'))
            # Extract real values of target variable from prediction data
            real_outputs = np.asarray(dp.get_column(predict_file, target))
            lh.log_gs_performance(
                current_model, results, datetimes, real_outputs, predictions,
                output_dir, run, file_name
            )
            # Delete model
            del ml_model


def train_model(train_file, predict_file, output_dir, model_size):
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
    list_epochs = [50]
    list_batches = [500]
    list_widths = [2, 4, 8]
    list_layers = [2, 3]
    # list_layers = [4, 5]
    # list_layers = [6, 7]
    # list_rates = [0.1, 0.01, 0.001, 0.0001]
    list_rates = [0.01]
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
    best_ml_model = None
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
                            ml_model = mdl.model(permutation, learning_rate)
                        else:
                            ml_model = mdl.big_model(permutation,
                                                     learning_rate)

                        # if load:
                        #     with file_io.FileIO('gs://nemsolutions-gcp-databucket/output/ex195/27037/ex195.h5', mode = 'r') as input:
                        #         ml_model.load_weights(input.name)
                        #     print("WEIGHTS: ", ml_model.get_weights())
                        # tbCallBack = keras.callbacks.TensorBoard(log_dir = '%s/%s' % (output_dir, train_file[-9:-3]), histogram_freq = 1, write_graph = True, write_images = True)

                        for train_index, test_index in kfold:
                            x, y = dp.reshape_data(x, y)
                            tensorboard = TensorBoard(log_dir =
                                                      ('gs://nemsolutions-gcp'
                                                       '-databucket/output/%s'
                                                       '/logs/%d'
                                                       % (output_dir.split('/')
                                                          [-1], run)),
                                                      histogram_freq = 0,
                                                      batch_size = batch_size,
                                                      write_graph = True,
                                                      write_grads = False,
                                                      write_images = False,
                                                      embeddings_freq = 0,
                                                      embeddings_layer_names = None,
                                                      embeddings_metadata = None)
                            ml_model, history = mdl.fit_model(ml_model,
                                                              x[train_index],
                                                              y[train_index],
                                                              epochs,
                                                              batch_size,
                                                              tensorboard)
                            temp_history_loss.append(history.history['loss'])
                            temp_history_val_loss.append(history.history
                                                         ['val_loss'])
                            final_train_scores.append(ml_model.evaluate(
                                x[train_index],
                                y[train_index]))
                            final_test_scores.append(ml_model.evaluate(
                                x[test_index],
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
                        file_name = train_file[-9:-4]
                        current_model = {'batch': batch_size,
                                         'epochs': epochs,
                                         'train_results': final_train_scores,
                                         'test_results': final_test_scores,
                                         'time': time,
                                         'layers': permutation,
                                         'learning_rate': learning_rate
                                         }

                        mdl.save_model(ml_model, '%s/%s' % (output_dir, run),
                                       file_name)
                        results, y, predictions = pred.predict(ml_model,
                                                               predict_file,
                                                               model_size,
                                                               batch_size)
                        datetimes = np.asarray(dp.get_column(predict_file,
                                                             'unixtime'))
                        real_power = np.asarray(dp.get_column(predict_file,
                                                              'WGENBearNDETemp'))

                        lh.log_gs_performance(current_model, results,
                                              datetimes, real_power,
                                              predictions, output_dir, run,
                                              file_name)
                        if (run == 1) | (np.mean(best_model['test_results']) >
                                                 np.mean(final_test_scores)):
                            # best_ml_model = ml_model
                            best_model = current_model
                        run += 1
                        del ml_model
        lh.log_gs_performance(best_model, results, datetimes, real_power,
                              predictions, output_dir, run, file_name)
        initial_time = t.time()
    # mdl.save_model(best_ml_model, output_dir, file_name)
    # lh.log_gs_performance(best_model, output_dir, file_name)

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
        '--job-name',
        help = 'Job name',
        required = True
    )
    parser.add_argument(
        '--batch-size',
        help = 'Batch size',
        required = True
    )
    parser.add_argument(
        '--model-size',
        help = 'Model size',
        required = True
    )
    parser.add_argument(
        '--epochs',
        help = 'Number of epochs',
        required = True
    )
    parser.add_argument(
        '--learning-rate',
        help = 'Learning rate',
        required = True
    )
    parser.add_argument(
        '--kfold-splits',
        help = 'Number of splits for KFold cross-validation',
        required = True
    )
    parser.add_argument(
        '--layers',
        help = 'Width of layers',
        required = True,
        nargs = '+'
    )
    args = parser.parse_args()
    arguments = args.__dict__
    numbers = re.compile('\d+(?:\.\d+)?')
    if arguments['custom'] == 'True':
        layers = [int(numbers.findall(width)[0]) for width in arguments['layers']]
        train_custom_model('gs://nemsolutions-gcp-databucket/input/filtered_modified_raw/train/27037.csv',
                           'gs://nemsolutions-gcp-databucket/input/filtered_modified_raw/test/27037.csv',
                           ('gs://nemsolutions-gcp-databucket/output/' + arguments['job_name']),
                    int(arguments['batch_size']), arguments['model_size'],
                    int(arguments['epochs']),
                    float(arguments['learning_rate']),
                    int(arguments['kfold_splits']), layers)
    else:
        train_model(arguments['train_file'], arguments['predict_file'],
                    arguments['job_dir'], arguments['model_size'])
