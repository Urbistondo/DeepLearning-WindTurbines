from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from util import data_preprocessor as dp
from local import model as mdl


def peak_predict(predict_file, model_size, batch):
    scaler = MinMaxScaler(feature_range = (0, 1))
    raw_x, raw_y = dp.get_data(predict_file, 1, model_size)
    norm_x, norm_y = dp.normalize_data(raw_x, raw_y, scaler)
    preprocessed_x, preprocessed_y = dp.reshape_data(norm_x, norm_y)
    #TODO - Proper models
    normal_model = mdl.load_model('../output/top/93/27037/weights.h5')
    peak_model = mdl.load_model('../output/top/93/27037/weights.h5')
    raw_predictions = normal_model.predict(preprocessed_x)
    raw_peak_predictions = peak_model.predict(preprocessed_x)
    reshaped_raw_predictions = [p[0][0] for p in raw_predictions]
    reshaped_raw_peak_predictions = [p[0][0] for p in raw_peak_predictions]
    denormalized_results = dp.denormalize_data(raw_y, reshaped_raw_predictions,
                                               scaler)
    denormalized_peak_results = dp.denormalize_data(
        raw_y, reshaped_raw_peak_predictions, scaler)
    combined_predictions = []
    buffer = []
    activated = False
    for index, value in enumerate(norm_y):
        if not activated:
            combined_predictions.append(denormalized_results[index])
            buffer.append(denormalized_results[index])
        else:
            combined_predictions.append(denormalized_peak_results[index])
            buffer.append(denormalized_peak_results[index])
        if index > 2:
            del buffer[0]
            for i in buffer:
                if i < 70:
                    activated = False
                    break
    normal_results = '\nRELATIVE PERFORMANCE: %s' \
                     '\nABSOLUTE PERFORMANCE: %s'\
                     % (mean_squared_error(norm_y, reshaped_raw_predictions),
                        mean_squared_error(raw_y, denormalized_results))
    peak_results = '\nRELATIVE PERFORMANCE: %s' \
                   '\nABSOLUTE PERFORMANCE: %s'\
                   % (mean_squared_error(norm_y, reshaped_raw_peak_predictions),
                      mean_squared_error(raw_y, denormalized_peak_results))
    print(normal_results)
    print(peak_results)
    return normal_results, peak_results, raw_y, denormalized_results, \
           denormalized_peak_results


def normal_predict(normal_model, predict_file, model_size, batch):
    scaler = MinMaxScaler(feature_range = (0, 1))
    raw_x, raw_y = dp.get_data(predict_file, 1, model_size)
    norm_x, norm_y = dp.normalize_data(raw_x, raw_y, scaler)
    preprocessed_x, preprocessed_y = dp.reshape_data(norm_x, norm_y)
    raw_predictions = normal_model.predict(preprocessed_x)
    reshaped_raw_predictions = [p[0][0] for p in raw_predictions]
    denormalized_results = dp.denormalize_data(raw_y, reshaped_raw_predictions,
                                               scaler)
    normal_predictions = []
    for index, value in enumerate(norm_y):
        normal_predictions.append(denormalized_results[index])
    normal_results = '\nRELATIVE PERFORMANCE: %s' \
             '\nABSOLUTE PERFORMANCE: %s'\
             % (mean_squared_error(norm_y, reshaped_raw_predictions),
                mean_squared_error(raw_y, denormalized_results))
    print(normal_results)
    return normal_results, raw_y, denormalized_results
