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

if __name__=='__main__':
    # load model from json
    with open('models/model_1.json', 'r') as f:
        model_json = json.load(f)
    model = model_from_json(model_json)

    # load model weights
    model.load_weights('models/model_1.h5')
    print('model and weights loaded')

    # load data: train set in inputs/outputs, test set
    X_train = load_from_pickle('data_Xy', 'X_train.pkl')
    y_train = load_from_pickle('data_Xy', 'y_train.pkl')
    test_df = load_from_pickle('data', 'data_test.pkl')
    test = test_df.values

    # compile and forecast
    def forecast(model, X_train, n_input):
        # grab the last week in training data
        input_x = X_train[-n_input:,0]
        # reshape into [1, n_input, 1]
        input_x = input_x.reshape((1, len(input_x), 1))
        # predict the next week
        model.compile(loss='mse', optimizer='adam')
        y_hat = model.predict(input_x, verbose=1)
        return y_hat

    y_hat = forecast(model, X_train, n_input=7)

    # evaluate on test set: first feature
    test = test[:,0]
    # walk-forward validation for each week
    n_input = 7
    true = []
    pred = []
    c = 0
    for i in range(len(test) - (2*n_input)):
        input = test[c:c + n_input]
        input = input.reshape(1, len(input), 1)
        y_true = test[c + n_input: c + 2*n_input]
        y_hat = model.predict(input, verbose=1)
        true.append(y_true)
        pred.append(y_hat.squeeze())
        print(c)
        c += 1

    # score predictions
    errors = []
    for y_true, y_hat in zip(true, pred):
        mse = mean_squared_error(y_true, y_hat)
        rmse = np.sqrt(mse)
        errors.append(rmse)

    save_to_pickle((true, pred, errors), 'output', 'output_1.pkl')



    #score = model.evaluate(X_train, y_train, verbose=1)
    #print('\n{}: {:0.2f}'.format(model.metrics_names[0], score[0]))
    #print('{}: {:0.2f}%'.format(model.metrics_names[1], score[1]*100))
