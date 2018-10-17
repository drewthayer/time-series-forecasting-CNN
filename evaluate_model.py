import pandas as pd
import numpy as np

from DataTools.pickle import save_to_pickle, load_from_pickle

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
    # 1D CNN imput shape: [n_samples, n_timesteps_per_sample, n_features]
    #                       e.g. [159 (week), 7(days), 1 (feature)] (or 8 features)
    train_data = train_data.reshape(int(train_data.shape[0]/7),7,train_data.shape[1])
    test_data = test_data.reshape(int(test_data.shape[0]/7),7,test_data.shape[1])
