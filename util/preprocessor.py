import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd


input_data = None
header = ['unixtime', 'STREQ', 'WTURNativeStatus', 'WROTPitchAngleAvgFreq',
          'WHDRGroupOilPressFreq', 'WROTPitchAngleAvgTravNeg',
          'WGENStatorCur', 'WGENSpeed', 'WROTSpeed', 'WNACWindSpeed',
          'WTURPower', 'WGENSpeedFreq', 'WCNVCoolantPress', 'WTRFPhase3Temp',
          'WGENRotorCur', 'WGENBearNDETemp', 'WTURReactivePowerAux',
          'WTURReactivePower', 'WROTPitchAngleSP', 'WYAWPressure',
          'WGENPhase3Temp', 'WTURCosPhi', 'WGENBearDETemp', 'WTRMBearTemp',
          'WGENPhase2Temp', 'WTRFPhase1Temp', 'WROTPitchAngleAvg',
          'WHDRGroupOilTemp', 'WGENStatorPower', 'WGENPhase1Temp',
          'WGENSlipRingTemp', 'WNACDirection', 'WCNVHeaterTempBott',
          'WNACWindDirection', 'WCNVHeaterTempSup', 'WNACWeatherAvai',
          'WNACWindSpeedStd10min', 'WHDRGroupOilPress',
          'WGENStatorReactivePower', 'WTRMOilTemp', 'WTURStatus',
          'WNACAmbTemp', 'WCNVCoolantTemp', 'WCNVNetVoltage',
          'WTURCur', 'WNACNacelleTemp', 'WTURPowerAux', 'WTRFPhase2Temp',
          'WGDCGridAvai', 'WNACWindSpeedNotFiltered',
          'WROTPitchAngleAvgTravPos', 'WHDRGroupOilPressTravNeg',
          'WHDRGroupOilPressTravPos', 'WTURPowerMed10min', 'WGENSpeedStd10min',
          'ESTR', 'WTURPowerStd10min', 'WROTPitchAngleAvgMax10min',
          'WNACWindDirectionCosMin10min', 'WROTSpeedMin10min',
          'WTURPowerMin10min'
          ]


# header = ['unixtime', 'WTRFPhase3Temp', 'WGENPhase3Temp',
#           'WTRMBearTemp', 'WGENPhase2Temp', 'WTRFPhase1Temp',
#           'WGENPhase1Temp', 'WCNVHeaterTempBott', 'WNACNacelleTemp',
#           'WTRFPhase2Temp', 'WTURStatus', 'WTURPowerMin10min',
#           'WROTPitchAngleAvgMax10min', 'WGENBearNDETemp'
#           ]


def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except:
        data = pd.DataFrame
    return data


def generate_ind_1(row):
    if (row['WTURStatus'] == 100) & \
            (row['WTURPowerMin10min'] > 0) & \
            (row['WROTPitchAngleAvgMax10min'] < 35):
        return 1
    return np.nan


def generate_ind_6(row):
    if (row['IND1'] == 1) & (row['WNACAmbTemp'] > -20) & \
            (row['WNACAmbTemp'] < 50):
        return row['WNACAmbTemp']
    return np.nan


def generate_ind_8(row):
    if (row['IND1'] == 1) & (row['WNACWindSpeed'] > 0) & \
            (row['WNACWindSpeed'] < 60):
        return row['WNACWindSpeed']
    return np.nan


def generate_ind_13(row):
    if (row['IND1'] == 1) & (row['WTURPower'] > 0) & \
            (row['WTURPower'] < 2500):
        return row['WTURPower']
    return np.nan


def generate_ind_228(row):
    if row['IND1'] == 1:
        return row['WNACWindDirection']


def generate_ind_230(row):
    return gap(row, 'IND228', 1000)


def generate_ind_263(row):
    return interval_count(row, 'IND230', 6000, 2)


def generate_ind_276(row):
    return gap(row, 'IND6', 1000)


def generate_ind_277(row):
    return interval_count(row, 'IND276', 6000, 2)


def generate_ind_672(row):
    return gap(row, 'IND13', 1000)


def generate_ind_673(row):
    return interval_count(row, 'IND672', 6000, 2)


def generate_ind_2822(row):
    return gap(row, 'IND8', 1000)


def generate_ind_2823(row):
    return interval_count(row, 'IND2822', 6000, 2)


def generate_ind_peak(row):
    if (row['WGENBearNDETemp'] > 72) & (row['WGENBearNDETemp'] < 120):
        return 1


def create_conditions(output_dir):
    global input_data
    input_data['IND1'] = input_data.apply(generate_ind_1, axis = 1)
    # input_data['IND6'] = input_data.apply(generate_ind_6, axis = 1)
    # input_data['IND8'] = input_data.apply(generate_ind_8, axis = 1)
    # input_data['IND13'] = input_data.apply(generate_ind_13, axis = 1)
    # input_data['IND228'] = input_data.apply(generate_ind_228, axis = 1)
    # input_data['IND230'] = input_data.apply(generate_ind_230, axis = 1)
    # input_data['IND263'] = input_data.apply(generate_ind_263, axis = 1)
    # input_data['IND276'] = input_data.apply(generate_ind_276, axis = 1)
    # input_data['IND277'] = input_data.apply(generate_ind_277, axis = 1)
    # input_data['IND672'] = input_data.apply(generate_ind_672, axis = 1)
    # input_data['IND673'] = input_data.apply(generate_ind_673, axis = 1)
    # input_data['IND2822'] = input_data.apply(generate_ind_2822, axis = 1)
    # input_data['IND2823'] = input_data.apply(generate_ind_2823, axis = 1)
    # input_data2 = input_data[['unixtime', 'IND1', 'IND13','IND263','IND277',
    #                         'IND673','IND2823']]
    # save_csv(input_data2, '../input/og_raw', 'inds.csv')
    return input_data


def create_peak_conditions(output_dir):
    global input_data
    input_data['IND1'] = input_data.apply(generate_ind_1, axis = 1)
    input_data['INDPEAK'] = input_data.apply(generate_ind_peak, axis = 1)
    # input_data2 = input_data[['unixtime', 'IND1', 'IND13','IND263','IND277',
    #                         'IND673','IND2823']]
    # save_csv(input_data2, '../input/og_raw', 'inds.csv')
    return input_data


def filter_data(data):
    filtered_data = data.loc[data['IND1'] == 1]
    # filtered_data = filtered_data[['unixtime', 'WNACWindSpeed', 'WNACWindSpeed|TFi|',
    #                             'WNACAmbTemp', 'WTURPower']]
    return filtered_data


def filter_peak_data(data):
    filtered_data = data.loc[(data['IND1'] == 1) & (data['INDPEAK'] == 1)]
    # filtered_data = data.loc[data['INDPEAK'] == 1]
    # filtered_data = filtered_data[['unixtime', 'WNACWindSpeed', 'WNACWindSpeed|TFi|',
    #                             'WNACAmbTemp', 'WTURPower']]
    return filtered_data


def filter(data):
    return data.dropna()


def gap(row, indicator, size):
    global input_data
    row_index = input_data.index.get_loc(row.name)
    if row_index >= size:
        aux = input_data.iloc[row_index - size]
        return aux[indicator]
    return np.nan


def interval_count(row, indicator, size, value):
    global input_data
    index = input_data.index.get_loc(row.name)
    if index >= size:
        aux = input_data[(index - size):index]
        return len(aux[aux[indicator] == value])
    return 0


def find_indexes(initial_date, final_date):
    global input_data
    dates = input_data['unixtime']
    initial_index = list(dates).index(np.int64(initial_date))
    final_index = list(dates).index(np.int64(final_date))
    # initial_not_found = True
    # final_not_found = True
    # while initial_not_found:
    #     try:
    #         initial_index = list(dates).index(np.int64(initial_date))
    #     except ValueError:
    #         initial_date += 600
    #         print(initial_date)
    #     else:
    #         initial_not_found = False
    #     print('Well, fuck me')
    # while final_not_found:
    #     try:
    #         final_index = list(dates).index(np.int64(final_date))
    #     except ValueError:
    #         final_date += 600
    #     else:
    #         final_not_found = False
    return initial_index, final_index


def save_csv(data, file_path, file_name):
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    data.to_csv('%s/%s' % (file_path, file_name), sep = ',', index = False)


def preprocess(input_dir, output_dir, initial_date, final_date, mode):
    for f in os.listdir(input_dir):
        print('Preprocessing %s' % f)
        global input_data
        global header
        input_data = read_data('%s/%s' % (input_dir, f))
        initial_timestamp = datetime.strptime(initial_date,
                                              '%d/%m/%Y').timestamp()
        final_timestamp = datetime.strptime(final_date,
                                            '%d/%m/%Y').timestamp()
        initial_index, final_index = find_indexes(initial_timestamp,
                                                  final_timestamp)
        input_data = input_data[initial_index:final_index + 1]
        input_data = create_conditions(output_dir)
        input_data = filter_data(input_data)
        input_data = input_data[header]
        input_data = input_data.dropna(axis = 'columns', how = 'all')
        header = input_data.columns
        input_data = filter(input_data)
        save_csv(input_data, '%s/%s' % (output_dir, mode), f)
        print('Successfully preprocessed: %s' % f)


def preprocess_peaks(input_dir, output_dir, initial_date, final_date, mode):
    for f in os.listdir(input_dir):
        print('Preprocessing %s' % f)
        global input_data
        global header
        input_data = read_data('%s/%s' % (input_dir, f))
        initial_timestamp = datetime.strptime(initial_date,
                                              '%d/%m/%Y').timestamp()
        final_timestamp = datetime.strptime(final_date,
                                            '%d/%m/%Y').timestamp()
        initial_index, final_index = find_indexes(initial_timestamp,
                                                  final_timestamp)
        input_data = input_data[initial_index:final_index + 1]
        input_data = create_peak_conditions(output_dir)
        input_data = filter_peak_data(input_data)
        input_data = input_data[header]
        input_data = input_data.dropna(axis = 'columns', how = 'all')
        header = input_data.columns
        input_data = filter(input_data)
        save_csv(input_data, '%s/%s' % (output_dir, mode), f)
        print('Successfully preprocessed: %s' % f)


def convert_to_gz(directory):
    for f in os.listdir(directory):
        print('Converting to .gz: %s' % f)
        og_name = str(directory + '/' + f)
        if not os.path.isfile(og_name): continue
        newname = str(og_name + '.gz')
        os.rename(og_name, newname)
        print('Successfully converted %s' % f)


def extract_csv_and_delete(directory):
    for f in os.listdir(directory):
        if (f[-3:] == '.gz'):
            print('Extracting: %s' % f)
            with open(directory + '/' + f, mode = 'r') as input_file:
                with open(directory + '/' + f[:-2] + 'csv', mode = 'w') as output_file:
                    wr = csv.writer(output_file, delimiter = ',', lineterminator = '\n')
                    for row in input_file.read().split('\n'):
                        wr.writerow(row.split(','))
            os.remove(directory + '/' + f)
        print('Successfully converted %s' % f)


def combine_csvs(input_directory, output_directory, file_name):
    first = True
    final_df = pd.DataFrame
    with open('../input/final/combined.csv', mode = 'w') as output_file:
        for f in os.listdir('../input/new'):
            print('Copying %s' % f)
            with open('%s/%s' % ('../input/new', f)) as input_file:
                if first:
                    final_df = read_data(input_file)
                    first = False
                else:
                    n_df = read_data(input_file)
                    final_df = pd.merge(final_df, n_df, how = 'outer')
        final_df.to_csv(output_file)
        print('Succesfully combined all files')


def run():
    input_dir = '../input/raw'
    # output_dir = '../input/preprocessed'
    output_dir = '../input/preprocessed_peaks'
    initial_train_date = '1/3/2015'
    final_train_date = '30/6/2016'
    initial_predict_date = '1/7/2016'
    final_predict_date = '30/9/2016'
    preprocess_peaks(input_dir, output_dir, initial_train_date, final_train_date, 'train')
    preprocess_peaks(input_dir, output_dir, initial_predict_date, final_predict_date, 'test')
    # preprocess(input_dir, output_dir, initial_train_date, final_train_date, 'train')
    # preprocess(input_dir, output_dir, initial_predict_date, final_predict_date, 'test')


# run()
