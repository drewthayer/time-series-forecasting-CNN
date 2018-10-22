import numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import model_from_json

from DataTools.pickle import save_to_pickle, load_from_pickle

if __name__=='__main__':
    # load model from json
    with open('models/model_1.json', 'r') as f:
        model_json = json.load(f)
    model = model_from_json(model_json)

    # load model weights
    model.load_weights('models/model_1.h5')
    print('model and weights loaded')

    # load data
    X_train = load_from_pickle('data_Xy', 'X_train.pkl')
    y_train = load_from_pickle('data_Xy', 'y_train.pkl')

    # compile
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # evaluate
    score = model.evaluate(X_train, y_train, verbose=1)
    print('\n{}: {:0.2f}'.format(model.metrics_names[0], score[0]))
    print('{}: {:0.2f}%'.format(model.metrics_names[1], score[1]*100))
