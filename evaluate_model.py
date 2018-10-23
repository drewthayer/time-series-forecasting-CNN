import numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error

from DataTools.pickle import save_to_pickle, load_from_pickle

def forecast(model, X_train, n_input):
    # grab the last week in training data
    input_x = X_train[-n_input:,0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # predict the next week
    model.compile(loss='mse', optimizer='adam')
    y_hat = model.predict(input_x, verbose=1)
    return y_hat

def walk_forward_validation(data, model, n_input):
    ''' walk-forward validation
        advances one unit at a time for a fixed set (e.g. daily, by week)
        inputs
            data        np array    must be shape that model has been trained on
            model       keras model with .predict() method and verbose parameter
            n_input     int         number of units to include in each input set
        outputs
            true        list        true values
            pred        list        predicted values, same length as true
        caveats
            data must match shape that model was trained on
            model must be trained on same n_input

    '''
    true = []
    pred = []
    c = 0
    for i in range(len(data) - (2*n_input)):
        input = data[c:c + n_input]
        input = input.reshape(1, len(input), 1)
        y_true = data[c + n_input: c + 2*n_input]
        y_hat = model.predict(input, verbose=1)
        true.append(y_true)
        pred.append(y_hat.squeeze()) # drop one dimension
        print(c)
        c += 1

    return true, pred

def calc_rmse_error(true, pred):
    ''' calc mse error from lists of true and predicted values
        uses sklearn.metrics.mean_squared_error()
        input lists can contain ints or arrays'''
    errors = []
    for y_true, y_hat in zip(true, pred):
        mse = mean_squared_error(y_true, y_hat)
        rmse = np.sqrt(mse)
        errors.append(rmse)
    return np.array(errors)

if __name__=='__main__':
    # load model and compile
    with open('models/model_1.json', 'r') as f:
        model_json = json.load(f)
    model = model_from_json(model_json)
    model.compile(loss='mse', optimizer='adam')

    # load model weights
    model.load_weights('models/model_1.h5')
    print('model and weights loaded')

    # load data: train set in inputs/outputs, test set
    X_train = load_from_pickle('data_Xy', 'X_train.pkl')
    y_train = load_from_pickle('data_Xy', 'y_train.pkl')
    test_df = load_from_pickle('data', 'data_test.pkl')
    test = test_df.values

    # evaluate on test set: univariate
    feat_col = 0
    test = test[:,feat_col]
    true, pred = walk_forward_validation(test, model, n_input=7)

    # score predictions, save to file
    errors = calc_rmse_error(true, pred)
    save_to_pickle((true, pred, errors), 'output', 'output_1.pkl')
