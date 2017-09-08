import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd

# Global variable to hold data
input_data = None


# Determine the columns to be present in resulting CSV files
header = ['unixtime', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'STREQ',
          'WTURNativeStatus', 'WROTPitchAngleAvgFreq',
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


def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except:
        data = pd.DataFrame
    return data


'''Generate Active Status indicator'''


def generate_ind_1(row):
    if (row['WTURStatus'] == 100) & \
            (row['WTURPowerMin10min'] > 0) & \
            (row['WROTPitchAngleAvgMax10min'] < 35):
        return 1
    return np.nan


'''Extract year from unix time'''


def generate_year(row):
    return datetime.fromtimestamp(row['unixtime']).year


'''Extract month from unix time'''


def generate_month(row):
    return datetime.fromtimestamp(row['unixtime']).month


'''Extract day from unix time'''


def generate_day(row):
    return datetime.fromtimestamp(row['unixtime']).day


'''Extract hour from unix time'''


def generate_hour(row):
    return datetime.fromtimestamp(row['unixtime']).hour


'''Extract minute from unix time'''


def generate_minute(row):
    return datetime.fromtimestamp(row['unixtime']).minute


'''Generate columns based on conditions'''


def create_conditions():
    global input_data
    # The condition is applied to every individual row and columns are added to
    # every individual row
    input_data['IND1'] = input_data.apply(generate_ind_1, axis = 1)
    input_data['MONTH'] = input_data.apply(generate_month, axis = 1)
    input_data['DAY'] = input_data.apply(generate_day, axis = 1)
    input_data['HOUR'] = input_data.apply(generate_hour, axis = 1)
    input_data['MINUTE'] = input_data.apply(generate_minute, axis = 1)
    return input_data


'''Drop rows whose rows don't meet criteria'''


def filter_data(data):
    filtered_data = data.loc[data['IND1'] == 1]
    return filtered_data


'''Drop rows containing null values'''


def filter(data):
    return data.dropna()


'''Find indexes inside dataframe corresponding to provided dates in unix time'''


def find_indexes(initial_date, final_date):
    global input_data
    dates = input_data['unixtime']
    initial_index = list(dates).index(np.int64(initial_date))
    final_index = list(dates).index(np.int64(final_date))
    return initial_index, final_index


'''Save CSV to desired path with desired file name'''


def save_csv(data, file_path, file_name):
    # If output directory doesn't exist, create it
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    data.to_csv('%s/%s' % (file_path, file_name), sep = ',', index = False)


'''Create CSV containing data between specified dates after applying conditions
   and filters and save it to a specific directory'''


def preprocess(input_dir, output_dir, initial_date, final_date, mode):
    for f in os.listdir(input_dir):
        print('Preprocessing %s' % f)
        global input_data
        global header
        # Read data from CSV
        input_data = read_data('%s/%s' % (input_dir, f))
        # Convert date in string format to timestamp
        initial_timestamp = datetime.strptime(
            initial_date, '%d/%m/%Y'
        ).timestamp()
        final_timestamp = datetime.strptime(
            final_date, '%d/%m/%Y'
        ).timestamp()
        # Obtain initial and final indexes
        initial_index, final_index = find_indexes(
            initial_timestamp, final_timestamp
        )
        # Get rows from initial to final index
        input_data = input_data[initial_index:final_index + 1]
        # Add relevant columns to dataframe
        input_data = create_conditions()
        # Apply conditions to filter the dataframe
        input_data = filter_data(input_data)
        # Get dataframe with only columns present in header list
        input_data = input_data[header]
        # Drop all columns containing null values
        input_data = input_data.dropna(axis = 'columns', how = 'all')
        # Update header list with non-dropped columns
        header = input_data.columns
        # Drop rows containing null values
        input_data = filter(input_data)
        # Save dataframe to CSV in specific output directory
        save_csv(input_data, '%s/%s' % (output_dir, mode), f)
        print('Successfully preprocessed: %s' % f)



'''Convert files in directory to .gz format'''


def convert_to_gz(input_dir):
    for f in os.listdir(input_dir):
        print('Converting to .gz: %s' % f)
        og_name = str(input_dir + '/' + f)
        if not os.path.isfile(og_name): continue
        newname = str(og_name + '.gz')
        os.rename(og_name, newname)
        print('Successfully converted %s' % f)


'''Extract and save CSV files from .gz files in directory  '''


def extract_csv_and_delete(input_dir):
    for f in os.listdir(input_dir):
        # Check if file extension coincides
        if f[-3:] == '.gz':
            print('Extracting: %s' % f)
            # Open .gz file
            with open(input_dir + '/' + f, mode = 'r') as input_file:
                # Open CSV file to save to
                with open(input_dir + '/' + f[:-2] + 'csv', mode = 'w')\
                        as output_file:
                    # Create CSV writer object
                    wr = csv.writer(
                        output_file, delimiter = ',', lineterminator = '\n'
                    )
                    # Read lines from .gz file and write to CSV
                    for row in input_file.read().split('\n'):
                        wr.writerow(row.split(','))
            # Remove .gz file
            os.remove(input_dir + '/' + f)
        print('Successfully converted %s' % f)


'''Combines all CSV files in a directory into a single CSV'''


def combine_csvs(input_directory, output_directory, file_name):
    first = True
    final_df = pd.DataFrame
    # Open destination CSV file
    with open('../input/final/combined.csv', mode = 'w') as output_file:
        for f in os.listdir('../input/new'):
            print('Copying %s' % f)
            # Open individual CSV file
            with open('%s/%s' % ('../input/new', f)) as input_file:
                # Append header to dataframe
                if first:
                    final_df = read_data(input_file)
                    first = False
                # Append entry to dataframe
                else:
                    n_df = read_data(input_file)
                    final_df = pd.merge(final_df, n_df, how = 'outer')
        # Save combined CSV
        final_df.to_csv(output_file)
        print('Succesfully combined all files')


def run():
    input_dir = '../input/raw'
    output_dir = '../input/preprocessed'
    initial_train_date = '1/3/2015'
    final_train_date = '30/6/2016'
    initial_predict_date = '1/7/2016'
    final_predict_date = '30/9/2016'
    preprocess(
        input_dir, output_dir, initial_train_date, final_train_date, 'train'
    )
    preprocess(
        input_dir, output_dir, initial_predict_date, final_predict_date, 'test'
    )


if __name__ == '__main__':
    run()
