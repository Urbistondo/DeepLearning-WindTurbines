import pandas as pd
import seaborn
from matplotlib import pyplot as plt

from util import preprocessor as p
import os


# '../output/trainings/101/27037/predictions.csv'
# '../input/preprocessed/test/27037.csv'
# '../output/trainings/101/27037/complete_predictions.csv'


# input_dir = '../input/preprocessed_peaks/train'
# for f in os.listdir(input_dir):
#     data = p.read_data('%s/%s' % (input_dir, f))
#     plt.plot(data['WGENBearNDETemp'])
#     plt.show()


# input_dir = '../input/preprocessed_peaks/without_year/train'
# dataframes = []
# for f in os.listdir(input_dir):
#     data = p.read_data('%s/%s' % (input_dir, f))
#     if type(data) == pd.DataFrame:
#         dataframes.append(data)
# full_data = pd.concat(dataframes)
# print(full_data)
# p.save_csv(full_data, '../input/preprocessed_peaks/without_year/train', 'full.csv')
#
# input_dir = '../input/preprocessed_peaks/without_year/test'
# dataframes.clear()
# for f in os.listdir(input_dir):
#     data = p.read_data('%s/%s' % (input_dir, f))
#     if type(data) == pd.DataFrame:
#         dataframes.append(data)
# full_data = pd.concat(dataframes)
# p.save_csv(full_data, '../input/preprocessed_peaks/without_year/test', 'full.csv')


# mydata1 = p.read_data('../input/preprocessed_peaks/train/full.csv')
# print(mydata1['WGENBearNDETemp'])
# mydata2 = p.read_data('../input/preprocessed_peaks/test/full.csv')
# print(mydata2['WGENBearNDETemp'])


# full = p.read_data('../input/preprocessed_combined/full.csv')
# full = full[['unixtime', 'WTRFPhase3Temp', 'WGENPhase3Temp',
#              'WTRMBearTemp', 'WGENPhase2Temp', 'WTRFPhase1Temp',
#              'WGENPhase1Temp', 'WCNVHeaterTempBott', 'WNACNacelleTemp',
#              'WTRFPhase2Temp', 'WTURStatus', 'WTURPowerMin10min', 'WROTPitchAngleAvgMax10min', 'WGENBearNDETemp']]
# p.save_csv(full, '../input/relevant', 'relevant.csv')


# my_predictions = p.read_data('../output/old_trainings/161/27037/predictions.csv')
# my_test_data = p.read_data('../input/correct/test/27037.csv')
# nem_predictions = p.read_data('../input/correct/NEM/NEM_results.csv')
# final_df = pd.merge(my_test_data, nem_predictions, on = 'unixtime')
# new_final_df = pd.merge(final_df, my_predictions, on = 'unixtime')
# final_final_df = new_final_df[['unixtime', 'WTURPower', 'MOD6', 'PREDICTED']]
# final_final_df = final_final_df.dropna()
# final_final_df['NEM'] = ((final_final_df['WTURPower'] - final_final_df['MOD6']) / final_final_df['WTURPower']) * 100
# final_final_df['URBISTONDO'] = ((final_final_df['WTURPower'] - final_final_df['PREDICTED']) / final_final_df['WTURPower']) * 100
# p.save_csv(final_final_df, '../input/correct', '161concatenation.csv')
# print(final_final_df)
#
#
# og_data = pd.read_csv('../input/og_raw/train/27037.csv')
# print(og_data)
# raw_data = pd.read_csv('../input/raw/27037.csv')
# data = pd.read_csv('../input/preprocessed2/inds.csv')
# print(raw_data['WTURStatus'])
# data['WTURStatus'] = raw_data['WTURStatus']
# data = data[['unixtime', 'IND1', 'WTURStatus', 'IND13', 'IND263', 'IND277', 'IND673', 'IND2823']]
# p.save_csv(data, '../input', 'inds.csv')
#
# counter = 0
# for index, row in data.iterrows():
#     if row['IND1'] == 1:
#         counter += 1
# print(data)
# print(counter)