from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from util import data_preprocessor as dp


'''Predict values for target variable using the provided trained model and
   the path to the predict file'''


def predict(model, predict_file, target):
    # Obtain scaler to denormalize predictions
    scaler = MinMaxScaler(feature_range = (0, 1))
    # Read input data from predict file
    raw_x, raw_y = dp.get_data(predict_file, 1, target)
    # Normalize data using scaler
    norm_x, norm_y = dp.normalize_data(raw_x, raw_y, scaler)
    # Reshape data to obtain 3-dimensional data to feed LSTM layers
    preprocessed_x, preprocessed_y = dp.reshape_data(norm_x, norm_y)
    # Reshape data to obtain 3-dimensional data to feed LSTM layers
    raw_predictions = model.predict(preprocessed_x)
    # Reshape data to obtain 1-dimensional vector from the predictions
    reshaped_raw_predictions = [p[0][0] for p in raw_predictions]
    # Denormalize predictions using scaler
    denormalized_results = dp.denormalize_data(
        reshaped_raw_predictions, scaler
    )
    performance_results = \
        '\nRELATIVE PERFORMANCE: %s' \
        '\nABSOLUTE PERFORMANCE: %s' \
        % (mean_squared_error(norm_y, reshaped_raw_predictions),
           mean_squared_error(raw_y, denormalized_results))
    return performance_results, denormalized_results
