import os
import csv
import numpy as np

from glob import glob
from tensorflow.python.lib.io.file_io import FileIO


def log_performance(
        model, results, datetimes, real_output, predictions, acc_history_loss,
        acc_history_val_loss, output_dir, job_name, file_name, target, run = -1
):
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
        if not os.path.isdir(directory):
            os.mkdir(directory)
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
    with open('%s/predictions.csv' % directory, mode = 'w') as output_file:
        wr = csv.writer(
            output_file, quoting = csv.QUOTE_MINIMAL, lineterminator = '\n'
        )
        header = ['DATETIME', 'REAL', 'PREDICTED']
        wr.writerow(header)
        rows = zip(datetimes, real_output, predictions)
        for row in rows:
            wr.writerow(row)
    with open('%s/loss.csv' % directory, mode = 'w') as output_file:
        wr = csv.writer(
            output_file, quoting = csv.QUOTE_MINIMAL, lineterminator = '\n'
        )
        header = ['TRAINING', 'VALIDATION']
        wr.writerow(header)
        rows = zip(acc_history_loss, acc_history_val_loss)
        for row in rows:
            wr.writerow(row)


def log_gs_performance(
        model, results, datetimes, real_output, predictions, output_dir, run,
        file_name
):
    directory = '%s/%s/%d' % (output_dir, file_name, run)
    with FileIO('%s/performance.txt' % (directory), mode = 'w') as output_file:
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
            '%s\nTRAIN: %f (%f) MSE /TEST: %f (%f) MSE'
            '\nLEARNING RATE: %f'
            '\nTIME: %f' %
            (results, np.mean(model['train_results']),
             np.std(model['train_results']), np.mean(model['test_results']),
             np.std(model['test_results']), model['learning_rate'],
             model['time'])
        )
    with FileIO('%s/predictions.csv' % directory, mode = 'w') as output_file:
        wr = csv.writer(output_file, quoting = csv.QUOTE_MINIMAL,
                        lineterminator = '\n')
        col1 = 'DATETIME'
        col2 = 'REAL'
        col3 = 'PREDICTED'
        first_row = zip([col1], [col2], [col3])
        wr.writerow(first_row)
        rows = zip(datetimes, real_output, predictions)
        for row in rows:
            wr.writerow(row)


def create_plotters(output_dir, job_name, file_name, target):
    directory = '%s/%d/%s/%s' % (output_dir, job_name, file_name, target)
    with open('%s/plot_predictions.py' % directory, mode = 'w') as output_file:
        output_file.write('import pandas as pd')
        output_file.write('\nfrom util import plot_handler as ph')
        output_file.write("\n\n\ndata = pd.read_csv('predictions.csv')")
        output_file.write(
            "\nph.show_plot(data.iloc[:,1], data.iloc[:,2], '%s',"
            " 'Model predictions')" % target
        )
    with open('%s/plot_loss.py' % directory, mode = 'w') as output_file:
        output_file.write('import pandas as pd')
        output_file.write('\nfrom util import plot_handler as ph')
        output_file.write("\n\n\ndata = pd.read_csv('loss.csv')")
        output_file.write(
            "\nph.show_plot(data.iloc[:,0], data.iloc[:,1], 'Loss',"
            " 'Model loss')"
        )