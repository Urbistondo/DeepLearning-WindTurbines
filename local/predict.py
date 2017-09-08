from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from util import data_preprocessor as dp
from local import model as mdl


def predict(model, predict_file, model_size, batch, target):
    scaler = MinMaxScaler(feature_range = (0, 1))
    raw_x, raw_y = dp.get_data(predict_file, 1, model_size, target)
    norm_x, norm_y = dp.normalize_data(raw_x, raw_y, scaler)
    preprocessed_x, preprocessed_y = dp.reshape_data(norm_x, norm_y)
    raw_predictions = model.predict(preprocessed_x)
    reshaped_raw_predictions = [p[0][0] for p in raw_predictions]
    denormalized_results = dp.denormalize_data(
        raw_y, reshaped_raw_predictions, scaler
    )
    # for index in range(len(raw_y)):
    #     if raw_y[index] < 75:
    #         denormalized_results[index] = raw_y[index]
    performance_results = \
        '\nRELATIVE PERFORMANCE: %s' \
        '\nABSOLUTE PERFORMANCE: %s' \
        % (mean_squared_error(norm_y, reshaped_raw_predictions),
           mean_squared_error(raw_y, denormalized_results))
    print(performance_results)
    return performance_results, raw_y, denormalized_results
