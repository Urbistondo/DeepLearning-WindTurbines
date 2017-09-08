import os
import csv
import numpy as np

from glob import glob
from tensorflow.python.lib.io.file_io import FileIO


'''Saves training results to TXT file locally'''


def log_performance(
        model, results, datetimes, real_output, predictions, acc_history_loss,
        acc_history_val_loss, output_dir, job_name, file_name, target, run = -1
):
    # Check if any of the necessary directories already exist. If not, create
    # them
    directory = '%s/%d' % (output_dir, job_name)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    directory = '%s/%s' % (directory, file_name)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    directory = '%s/%s' % (directory, target)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    # Identifies whether multiple trainings for the same train-predict files
    # are being run in order to create separate output directories for each
    if run > -1:
        directory = '%s/%d' % (directory, run)
        if not os.path.isdir(directory):
            os.mkdir(directory)
    # Save information about training to TXT file
    with open('%s/performance.txt' % directory, mode = 'w') as output_file:
        output_file.write(
            'BATCH: %s'
            '\nEPOCHS: %s'
            '\nLAYERS:' % (model['batch'], model['epochs'])
        )
        counter = 1
        for x in model['layers']:
            output_file.write('\n    LAYER %d: %s' % (counter, x))
            counter += 1
        output_file.write(
            '%s'
            '\nLEARNING RATE: %f'
            '\nTIME: %f'
            % (results, float(model['learning_rate']), model['time'])
        )
    # Save real and predicted values to CSV file
    with open('%s/predictions.csv' % directory, mode = 'w') as output_file:
        wr = csv.writer(
            output_file, quoting = csv.QUOTE_MINIMAL, lineterminator = '\n'
        )
        header = ['DATETIME', 'REAL', 'PREDICTED']
        wr.writerow(header)
        rows = zip(datetimes, real_output, predictions)
        for row in rows:
            wr.writerow(row)
    # Save training and validation loss across trainings to CSV file
    with open('%s/loss.csv' % directory, mode = 'w') as output_file:
        wr = csv.writer(
            output_file, quoting = csv.QUOTE_MINIMAL, lineterminator = '\n'
        )
        header = ['TRAINING', 'VALIDATION']
        wr.writerow(header)
        rows = zip(acc_history_loss, acc_history_val_loss)
        for row in rows:
            wr.writerow(row)


'''Saves training results to TXT file in Google Cloud Storage'''


def log_gs_performance(
        model, results, datetimes, real_output, predictions, acc_history_loss,
        acc_history_val_loss, output_dir, run, file_name, target
):
    directory = '%s/%s/%d' % (output_dir, file_name, run)
    # Save information about training to TXT file
    with FileIO('%s/performance.txt' % directory, mode = 'w') as output_file:
        output_file.write(
            '\nBATCH: %s'
            '\nEPOCHS: %s '
            '\nLAYERS:' % (model['batch'], model['epochs'])
        )
        counter = 1
        for x in model['layers']:
            output_file.write('\n    LAYER %d: %s' % (counter, x))
            counter += 1
        output_file.write(
            '%s'
            '\nLEARNING RATE: %f'
            '\nTIME: %f'
            % (results, float(model['learning_rate']), model['time'])
        )
    # Save real and predicted values to CSV file
    with FileIO('%s/predictions.csv' % directory, mode = 'w') as output_file:
        wr = csv.writer(
            output_file, quoting = csv.QUOTE_MINIMAL, lineterminator = '\n'
        )
        header = ['DATETIME', 'REAL', 'PREDICTED']
        wr.writerow(header)
        rows = zip(datetimes, real_output, predictions)
        for row in rows:
            wr.writerow(row)
    # Save training and validation loss across trainings to CSV file
    with FileIO('%s/loss.csv' % directory, mode = 'w') as output_file:
        wr = csv.writer(
            output_file, quoting = csv.QUOTE_MINIMAL, lineterminator = '\n'
        )
        header = ['TRAINING', 'VALIDATION']
        wr.writerow(header)
        rows = zip(acc_history_loss, acc_history_val_loss)
        for row in rows:
            wr.writerow(row)


'''Create Python scripts to plot training loss data and prediction data'''


def create_plotters(output_dir, job_name, file_name, target):
    directory = '%s/%d/%s/%s' % (output_dir, job_name, file_name, target)
    # Save predictions to Python script
    with open('%s/plot_predictions.py' % directory, mode = 'w') as output_file:
        output_file.write('import pandas as pd')
        output_file.write('\nfrom util import plot_handler as ph')
        output_file.write("\n\n\ndata = pd.read_csv('predictions.csv')")
        output_file.write(
            "\nph.show_plot(data.iloc[:,1], data.iloc[:,2], '%s',"
            " 'Model predictions')" % target
        )
    # Save loss data to Python script
    with open('%s/plot_loss.py' % directory, mode = 'w') as output_file:
        output_file.write('import pandas as pd')
        output_file.write('\nfrom util import plot_handler as ph')
        output_file.write("\n\n\ndata = pd.read_csv('loss.csv')")
        output_file.write(
            "\nph.show_plot(data.iloc[:,0], data.iloc[:,1], 'Loss',"
            " 'Model loss')"
        )