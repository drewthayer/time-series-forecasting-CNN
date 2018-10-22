import pandas as pd
import numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from DataTools.pickle import save_to_pickle, load_from_pickle

def build_staggered_Xy_1d(data, feature_idx, n_input=7, n_output=7):
    ''' feature_idx:    int     index of feature in array '''
    X, y = [],[]
    in_start = 0
    # step over timeseries: 'in' and 'out' series, lengths = n_input, n_output
    # series are contiguous so in_end = out_start
    for i in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_output
        if out_end < len(data): # prevent iterating too far
            x_input = data[in_start:in_end, feature_idx]
            x_input = x_input.reshape((len(x_input), 1)) # column vector
            X.append(x_input)
            y.append(data[in_end:out_end, feature_idx])
        # step forward one unit
        in_start += 1

    return np.array(X), np.array(y)

def build_model(X_train, y_train):
    # define parameters
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    return model

if __name__=='__main__':
    train_df = load_from_pickle('data','data_train.pkl')
    test_df = load_from_pickle('data','data_test.pkl')

    # TRANSFORM to np array
    # columns: ['Global_active_power', 'Global_reactive_power', 'Voltage',
    #   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
    #   'Sub_metering_3', 'Sub_metering_4']
    train_data = train_df.values
    test_data = test_df.values

    # TRANSFORM to CNN input shape
    # 1D CNN input shape: [n_samples, n_timesteps_per_sample, n_features]
    #                       e.g. [159 (week), 7(days), 1 (feature)] (or 8 features)
    train_data = train_data.reshape(int(train_data.shape[0]/7),7,train_data.shape[1])
    test_data = test_data.reshape(int(test_data.shape[0]/7),7,test_data.shape[1])

    # flatten back
    train_data_flat = train_data.reshape((train_data.shape[0]*train_data.shape[1], train_data.shape[2]))

    # convert timeseries into staggered inputs and outputs
    X_train, y_train = build_staggered_Xy_1d(train_data_flat, feature_idx=0, n_input=7, n_output=7)
    save_to_pickle(X_train, 'data_Xy', 'X_train.pkl')
    save_to_pickle(y_train, 'data_Xy', 'y_train.pkl')

    # train the model
    model = build_model(X_train, y_train)
    # fit network
    verbose, epochs, batch_size = 0, 20, 4
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # serialize and save model
    model_json = model.to_json()
    with open('models/model_1.json', 'w') as f:
        json.dump(model_json, f)

    # serialize weights to hdf5
    model.save_weights('models/model_1.h5')
    print('model and weights written to disk')
