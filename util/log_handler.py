import os
import csv
import numpy as np

from glob import glob
from tensorflow.python.lib.io.file_io import FileIO


def create_log(dir):
    result = [y.replace('\\', '/') for x in os.walk("%s" % dir) for y in glob(os.path.join(x[0], '*.txt'))]
    with open("./cloud/global_log.txt", mode = "a") as global_log:
        index = 1
        for x in result:
            with open(x, mode = "r") as trial_log:
                global_log.write('%s %s\n' % (index, trial_log.read()))
            index += 1


def log_performance(model, results, datetimes, real_power, predictions, output_dir, job_name,
                    file_name, target, run = -1):
    if not os.path.isdir('%s/%d' % (output_dir, job_name)):
        os.mkdir('%s/%d' % (output_dir, job_name))
    directory = '%s/%d/%s' % (output_dir, job_name, file_name)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    directory = '%s/%s' % (directory, target)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    if run != -1:
        directory = '%s/%d/%s/%d' % (output_dir, job_name, file_name, run)
        os.mkdir(directory)
    with open('%s/performance.txt' % directory,
              mode = 'w') as output_file:
        output_file.write('BATCH: %s'
                          '\nEPOCHS: %s'
                          '\nLAYERS:' % (model['batch'], model['epochs']))
        counter = 1
        for x in model['layers']:
            output_file.write('\n    LAYER %d: %s' % (counter, x))
            counter += 1
        output_file.write('%s\nTRAIN: %f (%f) MSE /TEST: %f (%f) MSE'
                          '\nLEARNING RATE: %f'
                          '\nTIME: %f' %
                          (results, np.mean(model['train_results']),
                           np.std(model['train_results']),
                           np.mean(model['test_results']),
                           np.std(model['test_results']),
                           float(model['learning_rate']),
                           model['time']))
    with open('%s/predictions.csv' % directory,
              mode = 'w') as output_file:
        wr = csv.writer(output_file, quoting = csv.QUOTE_MINIMAL, lineterminator = '\n')
        col1 = 'DATETIME'
        col2 = 'REAL'
        col3 = 'PREDICTED'
        first_row = zip([col1], [col2], [col3])
        wr.writerow(first_row)
        rows = zip(datetimes, real_power, predictions)
        for row in rows:
            wr.writerow(row)


def log_gs_performance(model, results, datetimes, real_power, predictions,
                       output_dir, run, file_name):
    with FileIO('%s/%s/%d/performance.txt' % (output_dir, file_name, run),
                mode = 'w') as output_file:
        output_file.write('\nBATCH: %s'
                          '\nEPOCHS: %s '
                          '\nLAYERS:' % (model['batch'], model['epochs']))
        counter = 1
        for x in model['layers']:
            output_file.write('\n    LAYER %d: %s' % (counter, x))
            counter += 1
        output_file.write('%s\nTRAIN: %f (%f) MSE /TEST: %f (%f) MSE'
                       '\nLEARNING RATE: %f'
                       '\nTIME: %f' %
                       (results,
                        np.mean(model['train_results']),
                        np.std(model['train_results']),
                        np.mean(model['test_results']),
                        np.std(model['test_results']),
                        model['learning_rate'],
                        model['time']))
    with FileIO('%s/%s/%d/predictions.csv' % (output_dir, file_name, run),
                mode = 'w') as output_file:
        wr = csv.writer(output_file, quoting = csv.QUOTE_MINIMAL, lineterminator = '\n')
        col1 = 'DATETIME'
        col2 = 'REAL'
        col3 = 'PREDICTED'
        first_row = zip([col1], [col2], [col3])
        wr.writerow(first_row)
        rows = zip(datetimes, real_power, predictions)
        for row in rows:
            wr.writerow(row)
