import numpy as np
import os


def validate_model_on_dataset(model, X, Y, scale_factor):
    channel_count = Y[0].shape[2]

    error_on_all = 0
    error_on_best = 0
    for channel in range(channel_count):
        for y_ground, y_pred in zip(Y, model.predict(X)):
            error = abs((np.sum(y_ground[:, :, channel]) - np.sum(y_pred[:, :, channel])) / scale_factor)
            error_on_all += error
            if channel > 2 and channel_count == 4:
                error_on_best += error

    error_on_all /= (len(Y) * channel_count)
    if channel_count == 4:
        error_on_best /= (len(Y) * 2)

    return error_on_all, error_on_best


def validate_model(model, X_train, Y_train, input_data_path, scale_factor):
    channel_count = Y_train[0].shape[2]
    X_test_seen = np.load(os.path.join(input_data_path, 'X_test_seen.npy'))
    Y_test_seen = np.load(os.path.join(input_data_path, 'Y_test_seen.npy'))
    X_test_unseen = np.load(os.path.join(input_data_path, 'X_test_unseen.npy'))
    Y_test_unseen = np.load(os.path.join(input_data_path, 'Y_test_unseen.npy'))

    error_on_all, error_on_best = validate_model_on_dataset(model, X_train, Y_train, scale_factor)
    print('Train MAE: {}'.format(error_on_all))
    if channel_count == 4:
        print('Train best MAE: {}'.format(error_on_best))

    error_on_all, error_on_best = validate_model_on_dataset(model, X_test_seen, Y_test_seen, scale_factor)
    print('Test seen MAE: {}'.format(error_on_all))
    if channel_count == 4:
        print('Test seen best MAE: {}'.format(error_on_best))

    error_on_all, error_on_best = validate_model_on_dataset(model, X_test_unseen, Y_test_unseen, scale_factor)
    print('Test unseen MAE: {}'.format(error_on_all))
    if channel_count == 4:
        print('Test seen best MAE: {}'.format(error_on_best))
