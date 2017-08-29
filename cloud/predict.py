import os

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from util import data_preprocessor as dp

# Avoid unnecessary warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def predict(ml_model, predict_file, model_size, batch):
    scaler = MinMaxScaler(feature_range = (0, 1))
    raw_x, raw_y = dp.get_data(predict_file, 1, model_size)
    norm_x, norm_y = dp.normalize_data(raw_x, raw_y, scaler)
    preprocessed_x, preprocessed_y = dp.reshape_data(norm_x, norm_y)
    raw_predictions = ml_model.predict(preprocessed_x)
    reshaped_raw_predictions = [p[0][0] for p in raw_predictions]
    denormalized_results = dp.denormalize_data(raw_y, reshaped_raw_predictions,
                                               scaler)
    predictions = []
    for index, value in enumerate(norm_y):
        predictions.append(denormalized_results[index])
    result = '\nRELATIVE PERFORMANCE: %s' \
             '\nABSOLUTE PERFORMANCE: %s'\
             % (mean_squared_error(norm_y, reshaped_raw_predictions),
                mean_squared_error(raw_y, denormalized_results))
    print(result)
    return result, raw_y, denormalized_results
